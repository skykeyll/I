"""Train the reproducible VDG-PPIS 4L/4VN/SH0.95 SaProt stacking model."""

import argparse
import json
import os
import time
from pathlib import Path
from typing import Dict, List, Tuple

os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np
import torch
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from torch.cuda.amp import GradScaler, autocast
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

from ppis_config import CONFIG, build_torch_generator, ensure_output_dir, set_seed
from ppis_data import ProteinGraphDataset, load_dataframe, single_item_collate
from ppis_losses import compute_loss, compute_metrics, sigmoid_probs
from ppis_model import PPISModel
from ppis_repro import sha256_file, write_runtime_audit_json
from ppis_splits import ensure_split_file, iter_split_indices

try:
    import joblib
except ImportError:
    joblib = None

SCRIPT_DIR = Path(__file__).resolve().parent

DEFAULT_EARLY_STOP_PATIENCE = 8
DEFAULT_HIDDEN_DIM = 64
DEFAULT_DROPOUT = 0.4
DEFAULT_WEIGHT_DECAY = 3e-4
DEFAULT_EGNN_LAYERS = 4
DEFAULT_IPR_WARMUP_EPOCHS = 0
DEFAULT_IPR_WARMUP_LR = 1e-3
FIXED_ENCODER_TYPE = "residue_dualequi"
FIXED_USE_IPR_VIRTUAL_NODES = True
FIXED_IPR_NUM_VIRTUAL_NODES = 4
FIXED_IPR_ASSIGNMENT_MODE = "balanced_sinkhorn"
FIXED_IPR_EVAL_HARD = False
FIXED_IPR_TEMPERATURE = 1.0
FIXED_IPR_SINKHORN_ITERATIONS = 20
FIXED_IPR_AUX_LOSS_WEIGHT = 1.0
FIXED_IPR_AUX_ENTROPY_WEIGHT = 0.001
FIXED_IPR_AUX_BALANCE_WEIGHT = 0.001
FIXED_IPR_AUX_COMPACT_WEIGHT = 0.01
FIXED_IPR_AUX_SMOOTH_WEIGHT = 0.01
FIXED_IPR_AUX_SCORE_L2_WEIGHT = 0.0005
FIXED_RESIDUE_USE_SH_BRANCH = True
FIXED_RESIDUE_USE_CROSS_SPACE = True
FIXED_RESIDUE_CROSS_SPACE_GATE = True
FIXED_RESIDUE_UPDATE_SH_NEIGHBORS = True
FIXED_RESIDUE_UPDATE_EU_NEIGHBORS = False
FIXED_RESIDUE_NUM_HEADS = 4
FIXED_RESIDUE_LMAX = 2
FIXED_RESIDUE_SH_CUTOFF = 0.95
FIXED_RESIDUE_EU_CUTOFF = 14.0
FIXED_RESIDUE_MAX_NEIGHBORS = 32
FIXED_LOSS_MODE = "bce_tversky"
FIXED_POS_WEIGHT_SCALE = 0.5
FIXED_TVERSKY_WEIGHT = 0.1
FIXED_TVERSKY_ALPHA = 0.4
FIXED_TVERSKY_BETA = 0.6

def apply_fixed_model_config() -> None:
    """Pin train2 to balanced Sinkhorn IPR virtual-node ResidueDualEqui."""
    CONFIG.encoder_type = FIXED_ENCODER_TYPE
    CONFIG.use_ipr_virtual_nodes = FIXED_USE_IPR_VIRTUAL_NODES
    CONFIG.ipr_num_virtual_nodes = FIXED_IPR_NUM_VIRTUAL_NODES
    CONFIG.ipr_assignment_mode = FIXED_IPR_ASSIGNMENT_MODE
    CONFIG.ipr_eval_hard = FIXED_IPR_EVAL_HARD
    CONFIG.ipr_temperature = FIXED_IPR_TEMPERATURE
    CONFIG.ipr_sinkhorn_iterations = FIXED_IPR_SINKHORN_ITERATIONS
    CONFIG.ipr_aux_loss_weight = FIXED_IPR_AUX_LOSS_WEIGHT
    CONFIG.ipr_aux_entropy_weight = FIXED_IPR_AUX_ENTROPY_WEIGHT
    CONFIG.ipr_aux_balance_weight = FIXED_IPR_AUX_BALANCE_WEIGHT
    CONFIG.ipr_aux_compact_weight = FIXED_IPR_AUX_COMPACT_WEIGHT
    CONFIG.ipr_aux_smooth_weight = FIXED_IPR_AUX_SMOOTH_WEIGHT
    CONFIG.ipr_aux_score_l2_weight = FIXED_IPR_AUX_SCORE_L2_WEIGHT
    CONFIG.residue_use_sh_branch = FIXED_RESIDUE_USE_SH_BRANCH
    CONFIG.residue_use_cross_space = FIXED_RESIDUE_USE_CROSS_SPACE
    CONFIG.residue_cross_space_gate = FIXED_RESIDUE_CROSS_SPACE_GATE
    CONFIG.residue_update_sh_neighbors = FIXED_RESIDUE_UPDATE_SH_NEIGHBORS
    CONFIG.residue_update_eu_neighbors = FIXED_RESIDUE_UPDATE_EU_NEIGHBORS
    CONFIG.residue_num_heads = FIXED_RESIDUE_NUM_HEADS
    CONFIG.residue_lmax = FIXED_RESIDUE_LMAX
    CONFIG.residue_sh_cutoff = FIXED_RESIDUE_SH_CUTOFF
    CONFIG.residue_eu_cutoff = FIXED_RESIDUE_EU_CUTOFF
    CONFIG.residue_max_neighbors = FIXED_RESIDUE_MAX_NEIGHBORS
    CONFIG.loss_mode = FIXED_LOSS_MODE
    CONFIG.pos_weight_scale = FIXED_POS_WEIGHT_SCALE
    CONFIG.tversky_weight = FIXED_TVERSKY_WEIGHT
    CONFIG.tversky_alpha = FIXED_TVERSKY_ALPHA
    CONFIG.tversky_beta = FIXED_TVERSKY_BETA


def count_trainable_parameters(model: torch.nn.Module) -> int:
    return sum(param.numel() for param in model.parameters() if param.requires_grad)


def enable_ipr_warmup_parameters(model: torch.nn.Module):
    original_requires_grad = {}
    warmup_params = []
    for name, param in model.named_parameters():
        original_requires_grad[name] = param.requires_grad
        trainable = name.startswith("egnn.ipr_score_proj.") or ".ipr_sampler.scorer." in name
        param.requires_grad_(trainable)
        if trainable:
            warmup_params.append(param)
    return original_requires_grad, warmup_params


def restore_requires_grad(model: torch.nn.Module, original_requires_grad: Dict[str, bool]) -> None:
    for name, param in model.named_parameters():
        param.requires_grad_(original_requires_grad.get(name, param.requires_grad))

def build_adamw_optimizer(model: torch.nn.Module) -> torch.optim.AdamW:
    decay_params = []
    no_decay_params = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if param.ndim <= 1 or name.endswith(".bias"):
            no_decay_params.append(param)
        else:
            decay_params.append(param)

    return torch.optim.AdamW(
        [
            {"params": decay_params, "weight_decay": CONFIG.weight_decay},
            {"params": no_decay_params, "weight_decay": 0.0},
        ],
        lr=CONFIG.lr,
    )


def move_batch_to_device(batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    output = {}
    for key, value in batch.items():
        if torch.is_tensor(value):
            output[key] = value.to(CONFIG.device)
        else:
            output[key] = value
    return output


def save_checkpoint(state: Dict, output_dir: str, filename: str) -> None:
    ensure_output_dir(output_dir)
    torch.save(state, os.path.join(output_dir, filename))

DEFAULT_SAPROT_DIR = os.path.join("Feature", "SaProt-WslFoldseek")
DEFAULT_OUTPUT_DIR = "runs"
DEFAULT_EXPERIMENT_NAME = ""
DEFAULT_SAPROT_EPOCHS = 40
BASE_MODEL_DIRNAME = "base_models"
META_MODEL_FILENAME = "meta_learner.joblib"
SUMMARY_FILENAME = "train_4l_4vn_sh095_summary.json"
DEFAULT_META_LEARNER = "hgb"
DEFAULT_META_POS_WEIGHT = 2.6
DEFAULT_META_LR_MAX_ITER = 200
DEFAULT_META_HGB_MAX_ITER = 50
DEFAULT_META_HGB_LEARNING_RATE = 0.10
DEFAULT_META_HGB_MAX_LEAF_NODES = 5
DEFAULT_META_HGB_L2_REGULARIZATION = 0.0
DEFAULT_SPLITS_PATH = SCRIPT_DIR / "splits" / "ppis_train335_seed2026_folds5.json"

def save_meta_model(model, path: str) -> None:
    if joblib is not None:
        joblib.dump(model, path)
        return
    import pickle

    with open(path, "wb") as f:
        pickle.dump(model, f)


def build_meta_learner(args, y_meta: np.ndarray):
    if args.meta_learner == "lr":
        model = LogisticRegression(
            class_weight={0: 1.0, 1: float(args.meta_pos_weight)},
            max_iter=args.meta_max_iter,
            solver="lbfgs",
        )
        config = {
            "type": "LogisticRegression",
            "class_weight": {0: 1.0, 1: float(args.meta_pos_weight)},
            "max_iter": int(args.meta_max_iter),
            "solver": "lbfgs",
            "C": 1.0,
        }
        return model, None, config

    if args.meta_learner == "hgb":
        model = HistGradientBoostingClassifier(
            max_iter=args.meta_max_iter,
            learning_rate=args.meta_hgb_learning_rate,
            max_leaf_nodes=args.meta_hgb_max_leaf_nodes,
            l2_regularization=args.meta_hgb_l2_regularization,
            early_stopping=False,
            random_state=args.seed,
        )
        config = {
            "type": "HistGradientBoostingClassifier",
            "positive_sample_weight": float(args.meta_pos_weight),
            "max_iter": int(args.meta_max_iter),
            "learning_rate": float(args.meta_hgb_learning_rate),
            "max_leaf_nodes": int(args.meta_hgb_max_leaf_nodes),
            "l2_regularization": float(args.meta_hgb_l2_regularization),
            "early_stopping": False,
            "random_state": int(args.seed),
        }
        sample_weight = np.where(y_meta == 1, float(args.meta_pos_weight), 1.0)
        return model, sample_weight, config

    raise ValueError(f"Unsupported meta learner: {args.meta_learner}")


def build_base_model_configs(esm_chunks: int, hidden_dim: int, all_feature_hidden_multiplier: float):
    if CONFIG.esm2_dim % esm_chunks != 0:
        raise ValueError(f"CONFIG.esm2_dim={CONFIG.esm2_dim} is not divisible by esm_chunks={esm_chunks}")

    chunk_dim = CONFIG.esm2_dim // esm_chunks
    configs = [
        {
            "name": "handcrafted",
            "filename": "handcrafted.pt",
            "feature_mode": "handcrafted",
            "input_dim": CONFIG.handcrafted_dim,
            "hidden_dim": hidden_dim,
        }
    ]
    for chunk_index in range(esm_chunks):
        configs.append(
            {
                "name": f"sap_chunk{chunk_index}",
                "filename": f"sap_chunk{chunk_index}.pt",
                "feature_mode": f"sap_chunk{chunk_index}",
                "chunk_index": chunk_index,
                "input_dim": chunk_dim,
                "hidden_dim": hidden_dim,
            }
        )
    configs.append(
        {
            "name": "all_features",
            "filename": "all_features.pt",
            "feature_mode": "all_features",
            "input_dim": CONFIG.handcrafted_dim + CONFIG.esm2_dim,
            "hidden_dim": max(1, int(round(hidden_dim * all_feature_hidden_multiplier))),
        }
    )
    return configs


def apply_runtime_config(args) -> None:
    apply_fixed_model_config()
    CONFIG.use_esm2 = True
    CONFIG.esm2_dir = args.esm2_dir
    CONFIG.esm2_dim = args.esm2_dim
    CONFIG.epochs = args.epochs
    CONFIG.hidden_dim = args.hidden_dim
    CONFIG.dropout = args.dropout
    CONFIG.weight_decay = args.weight_decay
    CONFIG.egnn_layers = args.egnn_layers
    CONFIG.lr = args.lr
    CONFIG.num_workers = args.num_workers
    CONFIG.early_stop_patience = args.early_stop_patience


def build_model(input_dim: int, edge_attr_dim: int, hidden_dim: int) -> PPISModel:
    apply_fixed_model_config()
    return PPISModel(
        input_dim=input_dim,
        edge_attr_dim=edge_attr_dim,
        hidden_dim=hidden_dim,
        egnn_layers=CONFIG.egnn_layers,
        dropout=CONFIG.dropout,
        use_ipr_virtual_nodes=CONFIG.use_ipr_virtual_nodes,
        ipr_num_virtual_nodes=CONFIG.ipr_num_virtual_nodes,
        ipr_temperature=CONFIG.ipr_temperature,
        ipr_sinkhorn_iterations=CONFIG.ipr_sinkhorn_iterations,
        ipr_eval_hard=CONFIG.ipr_eval_hard,
        ipr_scorer_context_layers=CONFIG.ipr_scorer_context_layers,
        ipr_virtual_message_layers=CONFIG.ipr_virtual_message_layers,
        ipr_aux_entropy_weight=CONFIG.ipr_aux_entropy_weight,
        ipr_aux_balance_weight=CONFIG.ipr_aux_balance_weight,
        ipr_aux_compact_weight=CONFIG.ipr_aux_compact_weight,
        ipr_aux_smooth_weight=CONFIG.ipr_aux_smooth_weight,
        ipr_aux_score_l2_weight=CONFIG.ipr_aux_score_l2_weight,
        encoder_type=CONFIG.encoder_type,
        residue_num_heads=CONFIG.residue_num_heads,
        residue_lmax=CONFIG.residue_lmax,
        residue_sh_cutoff=CONFIG.residue_sh_cutoff,
        residue_update_sh_neighbors=CONFIG.residue_update_sh_neighbors,
        residue_update_eu_neighbors=CONFIG.residue_update_eu_neighbors,
        residue_use_sh_branch=CONFIG.residue_use_sh_branch,
        residue_use_cross_space=CONFIG.residue_use_cross_space,
        residue_cross_space_gate=CONFIG.residue_cross_space_gate,
        residue_eu_cutoff=CONFIG.residue_eu_cutoff,
        residue_max_neighbors=CONFIG.residue_max_neighbors,
    ).to(CONFIG.device)


def build_loader(dataframe, psepos_path: str, shuffle: bool, seed: int, num_workers: int):
    dataset = ProteinGraphDataset(dataframe, psepos_path=psepos_path)
    generator = build_torch_generator(seed) if shuffle else None
    loader = DataLoader(
        dataset,
        batch_size=CONFIG.batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        generator=generator,
        collate_fn=single_item_collate,
    )
    return dataset, loader


def build_pos_weight_from_dataframe(dataframe) -> torch.Tensor:
    positive = 0
    total = 0
    for labels in dataframe["label"].tolist():
        labels_array = np.asarray(labels)
        positive += int(labels_array.sum())
        total += int(labels_array.size)
    negative = max(total - positive, 1)
    ratio = negative / max(positive, 1)
    scaled_ratio = 1.0 + (ratio - 1.0) * CONFIG.pos_weight_scale
    return torch.tensor([scaled_ratio], dtype=torch.float32, device=CONFIG.device)


def select_features(node_features: torch.Tensor, feature_mode: str, esm_chunks: int) -> torch.Tensor:
    node_features = node_features.to(CONFIG.device).float()
    if node_features.dim() == 3:
        node_features = node_features.squeeze(0)

    feature_dim = node_features.size(-1)
    handcrafted_dim = CONFIG.handcrafted_dim
    full_dim = CONFIG.handcrafted_dim + CONFIG.esm2_dim

    if feature_dim == handcrafted_dim:
        handcrafted = node_features
        esm2 = None
    elif feature_dim == full_dim:
        handcrafted = node_features[:, :handcrafted_dim]
        esm2 = node_features[:, handcrafted_dim:]
    else:
        raise ValueError(f"Unexpected node feature dim: {feature_dim}; expected {handcrafted_dim} or {full_dim}")

    if feature_mode == "handcrafted":
        return handcrafted
    if esm2 is None:
        raise ValueError(f"{feature_mode} requires CONFIG.use_esm2=True and residue language-model node features")
    if feature_mode == "all_features":
        return torch.cat([handcrafted, esm2], dim=-1)
    if feature_mode.startswith("sap_chunk"):
        chunk_index = int(feature_mode.replace("sap_chunk", ""))
        return torch.chunk(esm2, chunks=esm_chunks, dim=-1)[chunk_index]
    if feature_mode.startswith("saprot_chunk"):
        chunk_index = int(feature_mode.replace("saprot_chunk", ""))
        return torch.chunk(esm2, chunks=esm_chunks, dim=-1)[chunk_index]
    if feature_mode.startswith("esm_chunk"):
        chunk_index = int(feature_mode.replace("esm_chunk", ""))
        return torch.chunk(esm2, chunks=esm_chunks, dim=-1)[chunk_index]

    raise ValueError(f"Unsupported feature_mode: {feature_mode}")


def clone_state_dict(model: torch.nn.Module):
    return {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}


def make_checkpoint_config(sample: Dict[str, torch.Tensor], train_seed: int, base_config: Dict[str, object], args):
    return {
        **CONFIG.__dict__,
        "input_dim": int(base_config["input_dim"]),
        "raw_dataset_input_dim": int(sample["node_features"].size(-1)),
        "edge_attr_dim": int(sample["edge_attr"].size(-1)),
        "handcrafted_dim": int(CONFIG.handcrafted_dim),
        "esm2_dim": int(CONFIG.esm2_dim),
        "saprot_dim": int(CONFIG.esm2_dim),
        "saprot_dir": CONFIG.esm2_dir,
        "esm_chunks": int(args.esm_chunks),
        "saprot_chunks": int(args.esm_chunks),
        "feature_mode": base_config["feature_mode"],
        "base_model_name": base_config["name"],
        "hidden_dim": int(base_config["hidden_dim"]),
        "train_seed": int(train_seed),
        "train2_saprot_stacking": True,
    }


def run_feature_ipr_warmup_epoch(model, loader, base_config, optimizer, scaler, esm_chunks: int) -> Dict[str, float]:
    model.train()
    losses = []
    diagnostics = {
        "hard_empty": [],
        "hard_usage_min": [],
        "hard_usage_max": [],
        "soft_usage_min": [],
        "soft_usage_max": [],
        "score_entropy": [],
    }

    for batch in loader:
        batch = move_batch_to_device(batch)
        optimizer.zero_grad(set_to_none=True)
        features = select_features(batch["node_features"], base_config["feature_mode"], esm_chunks)

        with autocast(enabled=CONFIG.use_amp):
            loss, batch_diagnostics = model.ipr_warmup_loss(
                features,
                batch["pos"],
                batch["edge_index"],
                batch["edge_attr"],
            )

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(
            [param for group in optimizer.param_groups for param in group["params"]],
            CONFIG.grad_clip_norm,
        )
        scaler.step(optimizer)
        scaler.update()

        losses.append(float(loss.detach().cpu().item()))
        for key in diagnostics:
            if key in batch_diagnostics:
                diagnostics[key].append(float(batch_diagnostics[key]))

    metrics = {"loss": sum(losses) / max(len(losses), 1)}
    for key, values in diagnostics.items():
        if values:
            metrics[key] = sum(values) / len(values)
    return metrics


def warmup_feature_ipr_scorer(model, loader, base_config, warmup_epochs: int, warmup_lr: float, esm_chunks: int):
    if warmup_epochs <= 0 or not getattr(model, "use_ipr_virtual_nodes", False):
        return []

    original_requires_grad, warmup_params = enable_ipr_warmup_parameters(model)
    if not warmup_params:
        restore_requires_grad(model, original_requires_grad)
        return []

    optimizer = torch.optim.AdamW(warmup_params, lr=warmup_lr, weight_decay=CONFIG.weight_decay)
    scaler = GradScaler(enabled=CONFIG.use_amp)
    history = []
    try:
        for epoch in range(1, warmup_epochs + 1):
            metrics = run_feature_ipr_warmup_epoch(model, loader, base_config, optimizer, scaler, esm_chunks)
            history.append({"epoch": epoch, **metrics})
            print(
                f"[{base_config['name']} IPR warmup] Epoch {epoch:03d}/{warmup_epochs:03d} | "
                f"loss={metrics['loss']:.4f} | "
                f"hard_empty={metrics.get('hard_empty', 0.0):.2f} | "
                f"entropy={metrics.get('score_entropy', 0.0):.4f}"
            )
    finally:
        restore_requires_grad(model, original_requires_grad)

    return history


def run_feature_epoch(model, loader, pos_weight, base_config, esm_chunks: int, optimizer=None, scaler=None):
    is_train = optimizer is not None
    model.train() if is_train else model.eval()

    losses = []
    main_losses = []
    aux_losses = []
    y_true = []
    y_score = []

    for batch in loader:
        batch = move_batch_to_device(batch)
        if is_train:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(is_train):
            with autocast(enabled=CONFIG.use_amp):
                features = select_features(batch["node_features"], base_config["feature_mode"], esm_chunks)
                if is_train and CONFIG.ipr_aux_loss_weight > 0:
                    logits, aux_loss = model(
                        features,
                        batch["pos"],
                        batch["edge_index"],
                        batch["edge_attr"],
                        return_aux=True,
                    )
                else:
                    logits = model(features, batch["pos"], batch["edge_index"], batch["edge_attr"])
                    aux_loss = logits.new_zeros(())

                loss = compute_loss(
                    logits=logits,
                    targets=batch["labels"],
                    pos_weight=pos_weight,
                    focal_alpha=CONFIG.focal_alpha,
                    focal_gamma=CONFIG.focal_gamma,
                    focal_weight=CONFIG.focal_weight,
                    loss_mode=CONFIG.loss_mode,
                    tversky_alpha=CONFIG.tversky_alpha,
                    tversky_beta=CONFIG.tversky_beta,
                    tversky_weight=CONFIG.tversky_weight,
                    label_smoothing=CONFIG.label_smoothing,
                )
                main_loss = loss
                if is_train and CONFIG.ipr_aux_loss_weight > 0:
                    loss = loss + CONFIG.ipr_aux_loss_weight * aux_loss

            if is_train:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), CONFIG.grad_clip_norm)
                scaler.step(optimizer)
                scaler.update()

        losses.append(float(loss.detach().cpu().item()))
        main_losses.append(float(main_loss.detach().cpu().item()))
        aux_losses.append(float(aux_loss.detach().cpu().item()))
        y_score.extend(sigmoid_probs(logits).detach().cpu().tolist())
        y_true.extend(batch["labels"].detach().cpu().int().tolist())

    metrics = compute_metrics(y_true, y_score)
    metrics["loss"] = sum(losses) / max(len(losses), 1)
    metrics["main_loss"] = sum(main_losses) / max(len(main_losses), 1)
    metrics["aux_loss"] = sum(aux_losses) / max(len(aux_losses), 1)
    return metrics


def predict_logits(model, loader, pos_weight, base_config, esm_chunks: int):
    model.eval()
    logits_out = []
    labels_out = []
    y_score = []

    with torch.no_grad():
        for batch in loader:
            batch = move_batch_to_device(batch)
            features = select_features(batch["node_features"], base_config["feature_mode"], esm_chunks)
            logits = model(features, batch["pos"], batch["edge_index"], batch["edge_attr"]).view(-1)
            logits_out.append(logits.detach().cpu().numpy().reshape(-1, 1))
            labels = batch["labels"].detach().cpu().int().numpy().reshape(-1)
            labels_out.append(labels)
            y_score.extend(sigmoid_probs(logits).detach().cpu().tolist())

    logits_array = np.concatenate(logits_out, axis=0)
    labels_array = np.concatenate(labels_out, axis=0)
    metrics = compute_metrics(labels_array.tolist(), y_score)
    return logits_array, labels_array, metrics


def train_with_validation(
    base_config,
    fold: int,
    train_dataframe,
    valid_dataframe,
    run_dir: str,
    args,
):
    train_seed = int(args.seed + fold * 100 + len(base_config["name"]))
    set_seed(train_seed)

    train_dataset, train_loader = build_loader(
        train_dataframe,
        CONFIG.train_psepos,
        shuffle=True,
        seed=train_seed,
        num_workers=args.num_workers,
    )
    _, valid_loader = build_loader(
        valid_dataframe,
        CONFIG.train_psepos,
        shuffle=False,
        seed=train_seed,
        num_workers=0,
    )
    sample = train_dataset[0]
    checkpoint_config = make_checkpoint_config(sample, train_seed, base_config, args)
    checkpoint_config["fold"] = int(fold)
    checkpoint_config["cv_training"] = True

    model = build_model(
        input_dim=int(base_config["input_dim"]),
        edge_attr_dim=int(sample["edge_attr"].size(-1)),
        hidden_dim=int(base_config["hidden_dim"]),
    )
    warmup_history = warmup_feature_ipr_scorer(
        model,
        train_loader,
        base_config,
        args.ipr_warmup_epochs,
        args.ipr_warmup_lr,
        args.esm_chunks,
    )
    checkpoint_config["ipr_warmup_history"] = warmup_history
    parameter_count = count_trainable_parameters(model)
    checkpoint_config["parameter_count"] = parameter_count

    optimizer = build_adamw_optimizer(model)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = GradScaler(enabled=CONFIG.use_amp)
    pos_weight = build_pos_weight_from_dataframe(train_dataframe)

    best_epoch = 0
    best_valid_auprc = -1.0
    best_valid_auroc = 0.0
    best_state = clone_state_dict(model)
    history = []
    no_improve_epochs = 0

    for epoch in range(1, args.epochs + 1):
        epoch_start = time.time()
        train_metrics = run_feature_epoch(
            model,
            train_loader,
            pos_weight,
            base_config,
            args.esm_chunks,
            optimizer=optimizer,
            scaler=scaler,
        )
        scheduler.step()
        valid_metrics = run_feature_epoch(
            model,
            valid_loader,
            pos_weight,
            base_config,
            args.esm_chunks,
            optimizer=None,
            scaler=None,
        )
        epoch_seconds = time.time() - epoch_start
        history.append(
            {
                "epoch": epoch,
                "train": train_metrics,
                "valid": valid_metrics,
                "epoch_seconds": epoch_seconds,
            }
        )
        print(
            f"[{base_config['name']} fold {fold}] Epoch {epoch:03d}/{args.epochs:03d} | "
            f"train_loss={train_metrics['main_loss']:.4f} | train_AUPRC={train_metrics['auprc']:.4f} | "
            f"valid_loss={valid_metrics['main_loss']:.4f} | valid_AUPRC={valid_metrics['auprc']:.4f} | "
            f"valid_AUROC={valid_metrics['auroc']:.4f} | valid_MCC={valid_metrics['mcc']:.4f} | "
            f"seconds={epoch_seconds:.2f}"
        )

        if valid_metrics["auprc"] > best_valid_auprc:
            best_epoch = epoch
            best_valid_auprc = valid_metrics["auprc"]
            best_valid_auroc = valid_metrics["auroc"]
            best_state = clone_state_dict(model)
            no_improve_epochs = 0
        else:
            no_improve_epochs += 1
            if no_improve_epochs >= args.early_stop_patience:
                print(
                    f"[{base_config['name']} fold {fold}] Early stopping at epoch {epoch:03d}; "
                    f"best_epoch={best_epoch:03d}, best_valid_AUPRC={best_valid_auprc:.4f}"
                )
                break

    model.load_state_dict(best_state)
    fold_dir = os.path.join(run_dir, "cv", base_config["name"], f"fold_{fold}")
    ensure_output_dir(fold_dir)
    checkpoint_config["best_epoch"] = int(best_epoch)
    checkpoint_config["best_valid_auprc"] = float(best_valid_auprc)
    save_checkpoint(
        {
            "model_state_dict": best_state,
            "config": checkpoint_config,
            "epoch": best_epoch,
            "history": history,
            "train_seed": train_seed,
            "cv_training": True,
        },
        fold_dir,
        "train_best.pt",
    )
    with open(os.path.join(fold_dir, "train_history.json"), "w", encoding="utf-8") as f:
        json.dump(
            {
                "base_model": base_config,
                "fold": fold,
                "train_size": len(train_dataframe),
                "valid_size": len(valid_dataframe),
                "best_epoch": best_epoch,
                "best_valid_auprc": best_valid_auprc,
                "best_valid_auroc": best_valid_auroc,
                "parameter_count": parameter_count,
                "ipr_warmup_history": warmup_history,
                "history": history,
            },
            f,
            indent=2,
        )

    logits, labels, pred_metrics = predict_logits(model, valid_loader, pos_weight, base_config, args.esm_chunks)
    return {
        "best_epoch": int(best_epoch),
        "best_valid_auprc": float(best_valid_auprc),
        "best_valid_auroc": float(best_valid_auroc),
        "parameter_count": int(parameter_count),
        "checkpoint": os.path.join(fold_dir, "train_best.pt"),
        "logits": logits,
        "labels": labels,
        "oof_metrics": pred_metrics,
    }


def train_full_base_model(base_config, train_dataframe, final_epochs: int, output_dir: str, args):
    train_seed = int(args.seed + 10000 + len(base_config["name"]))
    set_seed(train_seed)
    train_dataset, train_loader = build_loader(
        train_dataframe,
        CONFIG.train_psepos,
        shuffle=True,
        seed=train_seed,
        num_workers=args.num_workers,
    )
    sample = train_dataset[0]
    checkpoint_config = make_checkpoint_config(sample, train_seed, base_config, args)
    checkpoint_config["final_training"] = True
    checkpoint_config["final_epochs"] = int(final_epochs)

    model = build_model(
        input_dim=int(base_config["input_dim"]),
        edge_attr_dim=int(sample["edge_attr"].size(-1)),
        hidden_dim=int(base_config["hidden_dim"]),
    )
    warmup_history = warmup_feature_ipr_scorer(
        model,
        train_loader,
        base_config,
        args.ipr_warmup_epochs,
        args.ipr_warmup_lr,
        args.esm_chunks,
    )
    checkpoint_config["ipr_warmup_history"] = warmup_history
    parameter_count = count_trainable_parameters(model)
    checkpoint_config["parameter_count"] = parameter_count

    optimizer = build_adamw_optimizer(model)
    scheduler = CosineAnnealingLR(optimizer, T_max=max(final_epochs, 1))
    scaler = GradScaler(enabled=CONFIG.use_amp)
    pos_weight = build_pos_weight_from_dataframe(train_dataframe)
    history = []

    for epoch in range(1, final_epochs + 1):
        train_metrics = run_feature_epoch(
            model,
            train_loader,
            pos_weight,
            base_config,
            args.esm_chunks,
            optimizer=optimizer,
            scaler=scaler,
        )
        scheduler.step()
        history.append({"epoch": epoch, "train": train_metrics})
        print(
            f"[{base_config['name']} full] Epoch {epoch:03d}/{final_epochs:03d} | "
            f"train_loss={train_metrics['main_loss']:.4f} | train_AUPRC={train_metrics['auprc']:.4f} | "
            f"train_AUROC={train_metrics['auroc']:.4f} | train_aux={train_metrics['aux_loss']:.4f}"
        )

    output_path = os.path.join(output_dir, base_config["filename"])
    save_checkpoint(
        {
            "model_state_dict": model.state_dict(),
            "config": checkpoint_config,
            "epoch": final_epochs,
            "history": history,
            "train_seed": train_seed,
            "final_training": True,
        },
        output_dir,
        base_config["filename"],
    )
    return {
        "name": base_config["name"],
        "checkpoint": output_path,
        "final_epochs": int(final_epochs),
        "parameter_count": int(parameter_count),
        "train_metrics": history[-1]["train"] if history else None,
    }


def train_saprot_stacking(args):
    apply_runtime_config(args)
    set_seed(args.seed)

    timestamp = time.strftime("%Y%m%d-%H%M%S")
    run_dir = os.path.join(args.output_dir, args.experiment_name, timestamp) if args.experiment_name else os.path.join(args.output_dir, timestamp)
    base_model_dir = os.path.join(run_dir, BASE_MODEL_DIRNAME)
    ensure_output_dir(base_model_dir)

    runtime_audit = write_runtime_audit_json(
        os.path.join(run_dir, "runtime_audit.json"),
        extra_files=[__file__],
    )

    train_dataframe = load_dataframe(CONFIG.train_dataset).reset_index(drop=True)
    base_configs = build_base_model_configs(args.esm_chunks, args.hidden_dim, args.all_feature_hidden_multiplier)
    fold_results = {config["name"]: [] for config in base_configs}
    oof_features = []
    oof_labels = []
    split_path = args.splits_path or os.path.join(run_dir, f"cv_splits_seed{args.seed}_folds{args.folds}.json")
    split_spec, split_path, split_created = ensure_split_file(
        train_dataframe,
        dataset_name=CONFIG.train_dataset,
        n_folds=args.folds,
        seed=args.seed,
        path=split_path,
    )

    print(
        f"========== Train2 SaProt stacking | proteins={len(train_dataframe)} | folds={args.folds} | "
        f"epochs={args.epochs} | saprot_chunks={args.esm_chunks} | saprot_dir={CONFIG.esm2_dir} | output={run_dir} =========="
    )
    print("Base models:", ", ".join(config["name"] for config in base_configs))
    print(f"Using fixed CV splits: {split_path} ({'created' if split_created else 'loaded'})")

    for fold, train_index, valid_index in iter_split_indices(split_spec, train_dataframe):
        print(f"\n========== Fold {fold} / {args.folds} ==========")
        fold_train = train_dataframe.iloc[train_index, :].reset_index(drop=True)
        fold_valid = train_dataframe.iloc[valid_index, :].reset_index(drop=True)
        fold_logits = []
        fold_labels = None

        for base_config in base_configs:
            print(f"\n----- Training {base_config['name']} -----")
            result = train_with_validation(base_config, fold, fold_train, fold_valid, run_dir, args)
            fold_results[base_config["name"]].append(
                {key: value for key, value in result.items() if key not in {"logits", "labels"}}
            )
            fold_logits.append(result["logits"])
            if fold_labels is None:
                fold_labels = result["labels"]
            elif not np.array_equal(fold_labels, result["labels"]):
                raise RuntimeError(f"OOF labels are not aligned in fold {fold}")

            if torch.cuda.is_available() and CONFIG.device.startswith("cuda"):
                torch.cuda.empty_cache()

        oof_features.append(np.concatenate(fold_logits, axis=1))
        oof_labels.append(fold_labels)

    x_meta = np.concatenate(oof_features, axis=0)
    y_meta = np.concatenate(oof_labels, axis=0)
    print(f"\nMeta learner training shape: X={x_meta.shape}, y={y_meta.shape}")

    meta_model, meta_sample_weight, meta_learner_config = build_meta_learner(args, y_meta)
    if meta_sample_weight is None:
        meta_model.fit(x_meta, y_meta)
    else:
        meta_model.fit(x_meta, y_meta, sample_weight=meta_sample_weight)
    meta_path = os.path.join(run_dir, META_MODEL_FILENAME)
    save_meta_model(meta_model, meta_path)
    torch.save(
        {
            "x_meta": x_meta,
            "y_meta": y_meta,
            "base_model_order": [config["name"] for config in base_configs],
            "base_model_configs": base_configs,
        },
        os.path.join(run_dir, "oof_meta_features.pt"),
    )
    print(f"Saved meta learner: {meta_path} | config={meta_learner_config}")

    final_results = []
    print("\n========== Train final base models on all training proteins ==========")
    for base_config in base_configs:
        best_epochs = [int(result["best_epoch"]) for result in fold_results[base_config["name"]]]
        final_epochs = max(1, int(round(sum(best_epochs) / max(len(best_epochs), 1))))
        print(f"\n----- Final {base_config['name']}: average best epoch {final_epochs} from {best_epochs} -----")
        final_results.append(train_full_base_model(base_config, train_dataframe, final_epochs, base_model_dir, args))
        if torch.cuda.is_available() and CONFIG.device.startswith("cuda"):
            torch.cuda.empty_cache()

    summary = {
        "run_dir": run_dir,
        "train_dataset": CONFIG.train_dataset,
        "train_psepos": CONFIG.train_psepos,
        "runtime_audit": runtime_audit,
        "args": vars(args),
        "splits": {
            "path": split_path,
            "sha256": sha256_file(split_path),
            "schema_version": split_spec.get("schema_version"),
            "strategy": split_spec.get("strategy"),
            "seed": split_spec.get("seed"),
            "n_folds": split_spec.get("n_folds"),
            "n_samples": split_spec.get("n_samples"),
        },
        "base_model_order": [config["name"] for config in base_configs],
        "base_model_configs": base_configs,
        "fold_results": fold_results,
        "meta_learner": meta_path,
        "meta_learner_config": meta_learner_config,
        "meta_training_shape": {"x": list(x_meta.shape), "y": list(y_meta.shape)},
        "final_results": final_results,
        "base_model_dir": base_model_dir,
    }
    with open(os.path.join(run_dir, SUMMARY_FILENAME), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(
        f"Train2 SaProt summary | run_dir={run_dir} | "
        f"meta_learner={meta_path} | base_model_dir={base_model_dir}"
    )


def parse_args():
    parser = argparse.ArgumentParser(description="Train train2 PPIS model with SaProt AA+3Di stacking.")
    parser.add_argument("--epochs", type=int, default=DEFAULT_SAPROT_EPOCHS, help="Max epochs for each CV base model")
    parser.add_argument("--folds", type=int, default=5, help="Number of protein-level folds")
    parser.add_argument("--seed", type=int, default=2026, help="KFold and training seed")
    parser.add_argument("--hidden_dim", type=int, default=DEFAULT_HIDDEN_DIM, help="Hidden dim for most base models")
    parser.add_argument(
        "--all_feature_hidden_multiplier",
        type=float,
        default=2.0,
        help="Hidden dim multiplier for the all-features base model",
    )
    parser.add_argument("--dropout", type=float, default=DEFAULT_DROPOUT, help="Dropout rate")
    parser.add_argument("--weight_decay", type=float, default=DEFAULT_WEIGHT_DECAY, help="AdamW weight decay")
    parser.add_argument("--lr", type=float, default=CONFIG.lr, help="AdamW learning rate")
    parser.add_argument("--egnn_layers", type=int, default=DEFAULT_EGNN_LAYERS, help="Number of dual-space layers")
    parser.add_argument("--num_workers", type=int, default=0, help="DataLoader workers; use 0 on Windows")
    parser.add_argument(
        "--early_stop_patience",
        type=int,
        default=DEFAULT_EARLY_STOP_PATIENCE,
        help="Stop CV base training after this many epochs without valid AUPRC improvement",
    )
    parser.add_argument(
        "--saprot_chunks",
        "--esm_chunks",
        dest="esm_chunks",
        metavar="SAPROT_CHUNKS",
        type=int,
        default=4,
        help="Split SaProt dimensions into this many chunks",
    )
    parser.add_argument(
        "--saprot_dir",
        "--esm2_dir",
        dest="esm2_dir",
        metavar="SAPROT_DIR",
        default=DEFAULT_SAPROT_DIR,
        help="Directory containing SaProt residue embedding .npy files",
    )
    parser.add_argument(
        "--saprot_dim",
        "--esm2_dim",
        dest="esm2_dim",
        metavar="SAPROT_DIM",
        type=int,
        default=1280,
        help="SaProt residue embedding dimension",
    )
    parser.add_argument(
        "--meta_learner",
        choices=["hgb", "lr"],
        default=DEFAULT_META_LEARNER,
        help="Meta learner for stacking logits",
    )
    parser.add_argument(
        "--meta_max_iter",
        type=int,
        default=None,
        help="Meta learner max_iter; defaults to 50 for hgb and 200 for lr",
    )
    parser.add_argument(
        "--meta_pos_weight",
        type=float,
        default=DEFAULT_META_POS_WEIGHT,
        help="Positive class/sample weight for meta learner",
    )
    parser.add_argument(
        "--meta_hgb_learning_rate",
        type=float,
        default=DEFAULT_META_HGB_LEARNING_RATE,
        help="HistGradientBoostingClassifier learning_rate",
    )
    parser.add_argument(
        "--meta_hgb_max_leaf_nodes",
        type=int,
        default=DEFAULT_META_HGB_MAX_LEAF_NODES,
        help="HistGradientBoostingClassifier max_leaf_nodes",
    )
    parser.add_argument(
        "--meta_hgb_l2_regularization",
        type=float,
        default=DEFAULT_META_HGB_L2_REGULARIZATION,
        help="HistGradientBoostingClassifier l2_regularization",
    )
    parser.add_argument(
        "--ipr_warmup_epochs",
        type=int,
        default=DEFAULT_IPR_WARMUP_EPOCHS,
        help="Feature-aware IPR scorer warm-up epochs before each base training",
    )
    parser.add_argument("--ipr_warmup_lr", type=float, default=DEFAULT_IPR_WARMUP_LR, help="IPR warm-up LR")
    parser.add_argument("--experiment_name", default=DEFAULT_EXPERIMENT_NAME, help="Subdirectory name")
    parser.add_argument("--output_dir", default=DEFAULT_OUTPUT_DIR, help="Directory for SaProt stacking runs")
    parser.add_argument(
        "--splits_path",
        default=None,
        help="Fixed CV split JSON to load; if missing, the file is created from --seed/--folds",
    )
    args = parser.parse_args()
    args.egnn_layers = 4
    if args.splits_path is None:
        args.splits_path = str(DEFAULT_SPLITS_PATH)
    if args.meta_max_iter is None:
        args.meta_max_iter = DEFAULT_META_HGB_MAX_ITER if args.meta_learner == "hgb" else DEFAULT_META_LR_MAX_ITER
    args.saprot_chunks = args.esm_chunks
    args.saprot_dir = args.esm2_dir
    args.saprot_dim = args.esm2_dim
    return args

if __name__ == "__main__":
    train_saprot_stacking(parse_args())
