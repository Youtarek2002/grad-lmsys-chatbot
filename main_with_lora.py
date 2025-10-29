# ruff: noqa: E402
import os

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
# os.environ["HF_HOME"] = "/path/to/fast/storage"

import argparse
import time
import shutil
import torch
import torch.nn.functional as F
import torch.distributed as dist
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    FullStateDictConfig,
    StateDictType,
)

from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    apply_activation_checkpointing,
)

from pathlib import Path

# Minimal replacements to avoid requiring detectron2 installation.
# We implement a tiny LazyConfig loader, instantiate helper, LRMultiplier, LRScheduler, and seed_all_rng.
import runpy
import importlib
from types import SimpleNamespace
import random
import numpy as np


class LazyConfig:
    @staticmethod
    def load(path):
        # If path is a .py file, execute it and return a SimpleNamespace of variables defined.
        ns = runpy.run_path(path)
        # Filter out builtins and modules
        filtered = {k: v for k, v in ns.items() if not k.startswith('__')}
        return SimpleNamespace(**filtered)

    @staticmethod
    def apply_overrides(cfg, overrides):
        # overrides: list of strings like "a.b=c"
        if not overrides:
            return cfg
        for o in overrides:
            if '=' not in o:
                continue
            key, val = o.split('=', 1)
            # navigate nested keys
            parts = key.split('.')
            obj = cfg
            for p in parts[:-1]:
                if hasattr(obj, p):
                    obj = getattr(obj, p)
                else:
                    # create nested namespace if missing
                    nested = SimpleNamespace()
                    setattr(obj, p, nested)
                    obj = nested
            last = parts[-1]
            # Try to evaluate val as Python literal, otherwise string
            try:
                evaluated = eval(val, {})
            except Exception:
                evaluated = val
            setattr(obj, last, evaluated)
        return cfg


def instantiate(spec):
    # If spec is callable, call it and return result.
    if callable(spec):
        try:
            return spec()
        except TypeError:
            # try calling without args
            return spec
    # If spec is a dict with _target_, try to import and construct
    if isinstance(spec, dict) and "_target_" in spec:
        target = spec["_target_"]
        module_name, attr = target.rsplit(".", 1)
        module = importlib.import_module(module_name)
        cls = getattr(module, attr)
        kwargs = {k: v for k, v in spec.items() if k != "_target_"}
        return cls(**kwargs)
    # If it's already an object, return as-is
    return spec


class LRMultiplier:
    def __init__(self, optimizer, multiplier=None, max_iter=None):
        self.optimizer = optimizer
        self.multiplier = multiplier
        self.max_iter = max_iter

    def step(self, *args, **kwargs):
        # placeholder: no-op or call underlying scheduler if present
        if hasattr(self.multiplier, "step"):
            return self.multiplier.step(*args, **kwargs)


class LRScheduler:
    @staticmethod
    def get_best_param_group_id(optimizer):
        # simple default
        return 0


def seed_all_rng(seed: int):
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    try:
        import torch.cuda
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:
        pass


from human_pref.logging import get_logger
from human_pref.utils import to_gpu


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("config")
    parser.add_argument("--load-from", default=None, type=str)
    parser.add_argument("--init-only", action="store_true")
    parser.add_argument("--eval-only", action="store_true")
    parser.add_argument("--no-log-file", action="store_true")
    parser.add_argument("--seed", type=int, default=-1)
    parser.add_argument("--output-root", default="../artifacts")
    parser.add_argument(
        "--opts",
        help="""
Modify config options at the end of the command, use "path.key=value".
        """.strip(),
        default=[],
        nargs=argparse.ZERO_OR_MORE,
    )
    parser.add_argument("--out", default=None, type=str)
    parser.add_argument("--use-small-gpu", action="store_true", help="Use 4-bit + LoRA mode for small GPUs (T4)")

    return parser.parse_args()


class LogLossBuffer:
    """Circular buffer for storing log loss values"""

    def __init__(self, size, device="cuda"):
        self.buffer = torch.zeros(size, device=device)
        self.size = size
        self.idx = 0
        self.num = 0

    def append(self, value):
        self.buffer[self.idx] = value
        self.idx = (self.idx + 1) % self.size
        self.num = min(self.num + 1, self.size)

    def mean(self):
        return self.buffer.sum().item() / self.num


@torch.no_grad()
def do_test(cfg, model):
    logger = get_logger("lmsys")
    logger.info("Evaluation start")

    val_loader = instantiate(cfg.dataloader.val)

    model.eval()
    from tqdm import tqdm

    rank = dist.get_rank()
    world_size = dist.get_world_size()
    if rank == 0:
        prog_bar = tqdm(val_loader)
    else:
        prog_bar = val_loader

    probs = []
    for batch in prog_bar:
        for micro_batch in batch:
            micro_batch = to_gpu(micro_batch)
            prob = model(micro_batch["input_ids"], micro_batch["cu_seqlens"]).softmax(
                dim=-1
            )
            gather_probs = [torch.zeros_like(prob) for _ in range(world_size)]
            dist.all_gather(gather_probs, prob)
            prob = torch.stack(gather_probs, dim=1).flatten(0, 1)
            probs.append(prob.data.cpu())

    result = torch.cat(probs, dim=0).numpy()
    # the last batch maybe padded to be divisible by world_size
    result = result[: len(val_loader.dataset)]

    logger.info("Evaluation prediction done")
    if not hasattr(val_loader.dataset, "evaluate"):
        eval_result = {"info": f"Not implemented for {type(val_loader.dataset)}"}
    else:
        eval_result = val_loader.dataset.evaluate(result)
    logger.info("Evaluation end")
    return result, eval_result


def save_checkpoint(model, optimizer, work_dir, checkpoint_path):
    save_policy = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
    with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT, save_policy):
        cpu_state = model.state_dict()
    if dist.get_rank() == 0:
        checkpoint = {
            "model": cpu_state,
            # "optimizer": optimizer.state_dict(),
        }
        torch.save(checkpoint, checkpoint_path)


def do_train(cfg, model):
    cfg.optimizer.params = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = instantiate(cfg.optimizer)

    train_loader = instantiate(cfg.dataloader.train)
    max_epochs = cfg.train.max_epochs
    lr_scheduler = LRMultiplier(
        optimizer,
        multiplier=instantiate(cfg.lr_multiplier),
        max_iter=max_epochs * len(train_loader),
    )
    best_param_group_id = LRScheduler.get_best_param_group_id(optimizer)

    logger = get_logger("lmsys")
    loss_history = LogLossBuffer(cfg.train.get("log_buffer_size", 100))
    total_updates = 0

    rank = dist.get_rank()
    fsdp_loss = torch.zeros(2).to(rank)

    clip_grad = cfg.train.get("clip_grad", True)
    for curr_epoch in range(max_epochs):
        model.train()
        for curr_iter, batch in enumerate(train_loader):
            total_batch_size = sum(micro_batch["batch_size"] for micro_batch in batch)
            fsdp_loss.zero_()
            for micro_batch in batch:
                micro_batch = to_gpu(micro_batch)
                logits = model(micro_batch["input_ids"], micro_batch["cu_seqlens"])
                loss = F.cross_entropy(logits, micro_batch["label"])
                fsdp_loss[0] += loss.detach() * micro_batch["batch_size"]
                fsdp_loss[1] += micro_batch["batch_size"]
                loss = loss * (micro_batch["batch_size"] / total_batch_size)
                loss.backward()

                dist.all_reduce(fsdp_loss, op=dist.ReduceOp.SUM)

            if clip_grad:
                grad_norm = model.clip_grad_norm_(1.0)
                grad_norm = grad_norm.item()
            else:
                grad_norm = 0
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

            loss_history.append(fsdp_loss[0] / fsdp_loss[1])
            total_updates += 1
            lr_scheduler.step()
            if total_updates % cfg.train.log_interval == 0:
                lr = optimizer.param_groups[best_param_group_id]["lr"]
                loss_val = loss_history.mean()
                max_mem_mb = torch.cuda.max_memory_allocated() / 1024.0 / 1024.0
                logger.info(
                    f"Epoch [{curr_epoch + 1}/{max_epochs}] Iter [{curr_iter + 1}/{len(train_loader)}]"
                    f" lr: {lr:.4e}, loss: {loss_val:.4f}, grad_norm: {grad_norm:.4f}, max_mem: {max_mem_mb:.0f}M"
                )

            # save every N updates
            if total_updates % cfg.train.checkpoint_interval == 0:
                checkpoint_path = (
                        Path(cfg.train.work_dir) / f"update_{total_updates}.pth"
                )
                logger.info(f"Save checkpoint: {checkpoint_path}")
                save_checkpoint(model, optimizer, cfg.train.work_dir, checkpoint_path)
                logger.info("Save checkpoint done.")
                dist.barrier()

        # end of epoch checkpoint
        checkpoint_path = Path(cfg.train.work_dir) / "update_last.pth"
        logger.info(f"Save checkpoint: {checkpoint_path}")
        save_checkpoint(model, optimizer, cfg.train.work_dir, checkpoint_path)
        logger.info("Save checkpoint done.")

        dist.barrier()

        # evaluate
        if (curr_epoch + 1) % cfg.train.get("eval_interval", 1) == 0:
            result, eval_result = do_test(cfg, model)
            if rank == 0:
                logger.info(f"Epoch {curr_epoch + 1} evaluation result: {eval_result}")
                torch.save(
                    result,
                    Path(cfg.train.work_dir) / f"result_epoch_{curr_epoch + 1}.pth",
                )


def setup(args):
    dist.init_process_group("nccl")
    torch.cuda.set_device(dist.get_rank())

    cfg = LazyConfig.load(args.config)
    # default work_dir
    cfg_path = Path(args.config)
    work_dir_root = Path(args.output_root)
    work_dir = str(work_dir_root / cfg_path.relative_to("configs/").with_suffix(""))
    cfg.train.work_dir = work_dir
    # override config
    cfg = LazyConfig.apply_overrides(cfg, args.opts)
    Path(cfg.train.work_dir).mkdir(parents=True, exist_ok=True)

    # dump config
    timestamp = time.strftime("%Y%m%d_%H%M%S", time.localtime())
    if not args.eval_only and dist.get_rank() == 0:
        # LazyConfig.save(cfg, str(Path(work_dir) / f"{timestamp}.yaml"))
        shutil.copy(args.config, Path(work_dir) / f"{timestamp}.py")

    # logger
    if args.eval_only or args.no_log_file:
        log_file = None
    else:
        log_file = Path(work_dir) / f"{timestamp}.log"
    logger = get_logger("lmsys", log_file=log_file)
    logger.info("Start")

    # seed
    if args.seed >= 0:
        seed = args.seed
    else:
        seed = cfg.train.get("seed", 0)
    seed_all_rng(seed)
    logger.info(f"Set random seed: {seed}")

    return cfg

def apply_lora_and_quant(cfg, model, logger):
    """
    Try to load HF model in 4-bit and attach LoRA adapters, or inject a simple LoRA wrapper
    into existing model if HF path isn't present. Returns the possibly-modified model.
    """
    try:
        # Lazy import heavy deps so normal runs don't require them
        from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
        from transformers import BitsAndBytesConfig, AutoModelForCausalLM, AutoTokenizer
        import bitsandbytes as bnb
        import torch.nn as nn
    except Exception as e:
        logger.warning(f"PEFT/bitsandbytes not available: {e}; skipping LoRA quant flow")
        return model

    # detect model path from cfg (common places)
    model_name_or_path = None
    if hasattr(cfg, "model"):
        model_obj = getattr(cfg, "model")
        model_name_or_path = getattr(model_obj, "name_or_path", None) or getattr(model_obj, "pretrained", None)

    if model_name_or_path:
        logger.info(f"Loading HF model from {model_name_or_path} in 4-bit mode and applying LoRA")
        bnb_cfg = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
        )
        tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=True)
        hf_model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            quantization_config=bnb_cfg,
            device_map="auto",
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
        )
        hf_model = prepare_model_for_kbit_training(hf_model)
        lora_cfg = LoraConfig(
            r=16,
            lora_alpha=32,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
        )
        hf_model = get_peft_model(hf_model, lora_cfg)
        logger.info("HF model loaded and LoRA applied.")
        return hf_model

    # Fallback: inject a simple LoRA wrapper into nn.Linear layers of the existing model
    logger.info("No HF model path found in cfg; injecting simple LoRA wrappers into existing model.")

    class LoRALinear(nn.Module):
        def __init__(self, orig_linear: nn.Linear, r=8, alpha=16, dropout=0.0):
            super().__init__()
            self.in_features = orig_linear.in_features
            self.out_features = orig_linear.out_features
            self.r = r
            self.alpha = alpha
            self.scaling = (self.alpha / self.r) if self.r > 0 else 1.0
            self.orig = orig_linear
            # freeze original
            self.orig.weight.requires_grad = False
            if self.orig.bias is not None:
                self.orig.bias.requires_grad = False
            if r > 0:
                self.A = nn.Parameter(torch.randn(r, self.in_features) * 0.01)
                self.B = nn.Parameter(torch.randn(self.out_features, r) * 0.01)
            else:
                self.register_parameter("A", None)
                self.register_parameter("B", None)
            self.dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()

        def forward(self, x):
            base = F.linear(x, self.orig.weight, self.orig.bias)
            if self.r > 0:
                lora_part = (self.dropout(x) @ self.A.t()) @ self.B.t()
                return base + lora_part * self.scaling
            return base

    def apply_lora_to_module(module, r=8, alpha=16, dropout=0.0, target_names=None):
        for name, child in list(module.named_children()):
            full_name = name
            if isinstance(child, nn.Linear):
                if (target_names is None) or any(t in full_name for t in target_names):
                    setattr(module, name, LoRALinear(child, r=r, alpha=alpha, dropout=dropout))
            else:
                apply_lora_to_module(child, r=r, alpha=alpha, dropout=dropout, target_names=target_names)

    try:
        apply_lora_to_module(model, r=8, alpha=16, dropout=0.05, target_names=None)
        logger.info("Injected LoRA wrappers into model.")
    except Exception as e:
        logger.warning(f"Failed to inject LoRA wrappers: {e}")

    return model


def clean_up():
    dist.destroy_process_group()

def main():
    args = parse_args()
    cfg = setup(args)
    logger = get_logger("lmsys")

    # ---- Detect if we’re on a T4 / small GPU ----
    small_gpu = torch.cuda.get_device_name(0).lower().find("t4") != -1 or bool(int(os.environ.get("USE_SMALL_GPU", "0")))

    if small_gpu:
        logger.info("Detected small GPU (T4). Using 4-bit quantization + LoRA fine-tuning mode.")
        try:
            from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
            from transformers import BitsAndBytesConfig, AutoModelForCausalLM, AutoTokenizer
            import bitsandbytes as bnb

            model_name_or_path = getattr(cfg.model, "name_or_path", None) or getattr(cfg.model, "pretrained", None)
            if model_name_or_path is None:
                raise ValueError("Model path not found in cfg.model")

            # Quantized 4-bit model load
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type='nf4',
                bnb_4bit_compute_dtype=torch.float16,
            )
            tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=True)
            model = AutoModelForCausalLM.from_pretrained(
                model_name_or_path,
                quantization_config=bnb_config,
                device_map="auto",
                torch_dtype=torch.float16,
                low_cpu_mem_usage=True,
            )
            model = prepare_model_for_kbit_training(model)

            # Apply LoRA adapters
            lora_cfg = LoraConfig(
                r=8,
                lora_alpha=16,
                target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
                lora_dropout=0.05,
                bias="none",
                task_type="CAUSAL_LM",
            )
            model = get_peft_model(model, lora_cfg)
            logger.info("✅ LoRA applied successfully.")
        except Exception as e:
            logger.error(f"Failed to initialize LoRA path: {e}")
            raise e
    else:
        # Regular path (for A100 or large GPU)
        model = instantiate(cfg.model)
        model = FSDP(
            model,
            auto_wrap_policy=cfg.fsdp.auto_wrap_policy,
            sharding_strategy=cfg.fsdp.sharding_strategy,
            device_id=torch.cuda.current_device(),
            mixed_precision=cfg.fsdp.mixed_precision,
        )
        apply_activation_checkpointing(model, auto_wrap_policy=cfg.fsdp.auto_wrap_policy)

    # --- Shared training/eval logic ---
    if args.init_only:
        init_path = Path(cfg.train.work_dir) / "initialized.pth"
        torch.save(model.state_dict(), init_path)
        logger.info(f"Saved initialized model: {init_path}")

    if cfg.train.get("cast_to_bf16", False):
        logger.info("Casting model to BF16")
        for p in model.parameters():
            p.data = p.data.to(torch.bfloat16)

    load_from = args.load_from or cfg.train.get("load_from", None)
    if load_from is not None and os.path.exists(load_from):
        checkpoint = torch.load(load_from, map_location="cpu")
        if "model" not in checkpoint:
            checkpoint = {"model": checkpoint}
        load_result = model.load_state_dict(checkpoint["model"], strict=False)
        logger.info(f"Loaded checkpoint from {load_from}: {load_result}")

    if args.eval_only:
        result, eval_result = do_test(cfg, model)
        logger.info(f"Evaluation result: {eval_result}")
        if args.out is not None:
            torch.save(result, args.out)
    else:
        do_train(cfg, model)

    clean_up()


if __name__ == "__main__":
    main()
