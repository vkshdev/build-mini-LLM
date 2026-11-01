import os
import time
import math
import argparse
from contextlib import nullcontext
from typing import Union
import torch
import wandb

from src.config import ModelConfig, TrainingConfig, WandbConfig
from src.tokenizer import get_tokenizer
from src.dataset import PretrainingBuffer, prepare_tinystories
from src.model import SmallLLM


def str2bool(v: Union[str, bool]) -> bool:
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    raise argparse.ArgumentTypeError(f"Boolean expected, got '{v}'.")


def parse_args():
    parser = argparse.ArgumentParser(description="Pre-train 28M LLM on TinyStories")
    parser.add_argument("--data_dir", type=str, default="data")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints")
    parser.add_argument("--resume", type=str2bool, default=False)
    parser.add_argument("--max_steps", type=int, default=10000)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=6e-4)
    parser.add_argument("--no_wandb", action="store_true")
    return parser.parse_args()


def get_lr(step: int, cfg: TrainingConfig) -> float:
    # warmup then cosine decay to min_lr
    if step < cfg.warmup_steps:
        return cfg.learning_rate * (step + 1) / (cfg.warmup_steps + 1)
    if step > cfg.max_steps:
        return cfg.min_lr
    decay_ratio = (step - cfg.warmup_steps) / (cfg.max_steps - cfg.warmup_steps)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return cfg.min_lr + coeff * (cfg.learning_rate - cfg.min_lr)


@torch.no_grad()
def estimate_loss(model: SmallLLM, buffer: PretrainingBuffer, eval_iters: int, batch_size: int, device: str, ctx) -> float:
    model.eval()
    losses = torch.zeros(eval_iters)
    for k in range(eval_iters):
        x, y = buffer.get_batch(batch_size, device)
        with ctx:
            _, loss = model(x, targets=y)
        losses[k] = loss.item()
    model.train()
    return losses.mean().item()


def main():
    args = parse_args()
    model_cfg = ModelConfig()
    train_cfg = TrainingConfig()
    wandb_cfg = WandbConfig()

    if args.max_steps:
        train_cfg.max_steps = args.max_steps
    if args.batch_size:
        train_cfg.batch_size = args.batch_size
    if args.lr:
        train_cfg.learning_rate = args.lr
    train_cfg.checkpoint_dir = args.checkpoint_dir
    train_cfg.resume = args.resume
    if args.no_wandb:
        wandb_cfg.enabled = False

    os.makedirs(train_cfg.checkpoint_dir, exist_ok=True)
    device = train_cfg.device
    device_type = 'cuda' if 'cuda' in device else 'cpu'

    # AMP setup
    amp_dtype = torch.float16 if train_cfg.dtype == 'float16' else torch.bfloat16
    ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=amp_dtype)
    scaler = torch.cuda.amp.GradScaler(enabled=(train_cfg.dtype == 'float16' and device_type == 'cuda'))

    print("=" * 60)
    print("Pre-training 28M LLM (Phase 1)")
    print(f"Device: {device.upper()} | Precision: {train_cfg.dtype}")
    print("=" * 60)

    # ensure data exists
    train_bin = os.path.join(args.data_dir, "train.bin")
    val_bin = os.path.join(args.data_dir, "val.bin")
    if not (os.path.exists(train_bin) and os.path.exists(val_bin)):
        prepare_tinystories(output_dir=args.data_dir)

    train_buffer = PretrainingBuffer(train_bin, block_size=model_cfg.block_size)
    val_buffer = PretrainingBuffer(val_bin, block_size=model_cfg.block_size)
    tokenizer = get_tokenizer()

    model = SmallLLM(model_cfg).to(device)
    print(f"Model params: {model.get_num_params(non_embedding=False) / 1e6:.2f}M")

    optimizer = model.configure_optimizers(
        weight_decay=train_cfg.weight_decay,
        learning_rate=train_cfg.learning_rate,
        betas=(train_cfg.beta1, train_cfg.beta2),
        device_type=device_type
    )

    start_step = 0
    best_val_loss = float('inf')
    latest_ckpt = os.path.join(train_cfg.checkpoint_dir, "checkpoint_latest.pt")

    # resume if requested
    if train_cfg.resume and os.path.exists(latest_ckpt):
        print(f"Resuming from: {latest_ckpt}")
        ckpt = torch.load(latest_ckpt, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        start_step = ckpt["step"] + 1
        best_val_loss = ckpt.get("best_val_loss", float('inf'))

    if wandb_cfg.enabled:
        wandb.init(
            project=wandb_cfg.project,
            name=wandb_cfg.run_name,
            entity=wandb_cfg.entity,
            tags=wandb_cfg.tags,
            config={**model_cfg.__dict__, **train_cfg.__dict__},
            resume="allow"
        )
        sample_table = wandb.Table(columns=["Step", "Prompt", "Generated Story"])

    test_prompts = [
        "Once upon a time in a sunny meadow,",
        "A little girl named Lily found a shiny silver key and",
        "The small brown dog barked happily because",
    ]

    t0 = time.time()
    accum_tokens = 0

    for step in range(start_step, train_cfg.max_steps):
        lr = get_lr(step, train_cfg)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        optimizer.zero_grad(set_to_none=True)
        loss_accum = 0.0

        # gradient accumulation
        for _ in range(train_cfg.gradient_accumulation_steps):
            x, y = train_buffer.get_batch(train_cfg.batch_size, device)
            with ctx:
                _, loss = model(x, targets=y)
                loss = loss / train_cfg.gradient_accumulation_steps
                loss_accum += loss.item()
            scaler.scale(loss).backward()
            accum_tokens += train_cfg.batch_size * model_cfg.block_size

        scaler.unscale_(optimizer)
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), train_cfg.grad_clip).item()
        scaler.step(optimizer)
        scaler.update()

        # logging
        if step % train_cfg.log_interval == 0:
            t1 = time.time()
            dt = t1 - t0
            tok_per_sec = accum_tokens / max(dt, 1e-5)
            accum_tokens = 0
            t0 = t1

            print(f"Step {step:5d}/{train_cfg.max_steps} | Loss: {loss_accum:.4f} | LR: {lr:.2e} | Grad: {grad_norm:.2f} | {tok_per_sec:,.0f} tok/s")
            if wandb_cfg.enabled:
                wandb.log({
                    "step": step,
                    "train/loss": loss_accum,
                    "train/lr": lr,
                    "train/grad_norm": grad_norm,
                    "system/tokens_per_sec": tok_per_sec,
                })

        # validation & generation sampling
        if step % train_cfg.eval_interval == 0 and step > 0:
            val_loss = estimate_loss(model, val_buffer, train_cfg.eval_iters, train_cfg.batch_size, device, ctx)
            ppl = math.exp(min(val_loss, 20.0))
            print(f"\n[VAL] Step {step:5d} | Val Loss: {val_loss:.4f} | Perplexity: {ppl:.2f}\n")

            if wandb_cfg.enabled:
                wandb.log({"step": step, "val/loss": val_loss, "val/perplexity": ppl})

            # save best model checkpoint
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_path = os.path.join(train_cfg.checkpoint_dir, "best_pretrained_28m.pt")
                torch.save({
                    "step": step,
                    "model_state_dict": {k: v.half() for k, v in model.state_dict().items()},
                    "best_val_loss": best_val_loss,
                    "config": model_cfg.__dict__,
                }, best_path)

                if wandb_cfg.enabled:
                    art = wandb.Artifact("pretrained-28m-best", type="model")
                    art.add_file(best_path)
                    wandb.log_artifact(art, aliases=["best", "latest"])

            # sample generation
            model.eval()
            for p in test_prompts:
                p_tokens = torch.tensor(tokenizer.encode(p), dtype=torch.long, device=device).unsqueeze(0)
                gen_tokens = model.generate(p_tokens, max_new_tokens=40, temperature=0.75, top_k=30)
                gen_text = tokenizer.decode(gen_tokens[0].tolist())
                print(f"  [Sample]: {gen_text}")
                if wandb_cfg.enabled:
                    sample_table.add_data(step, p, gen_text)
            print()
            model.train()

            if wandb_cfg.enabled:
                wandb.log({"samples/emergence_table": sample_table})

        # regular backup dump for disconnect recovery
        if step % train_cfg.save_interval == 0 and step > 0:
            torch.save({
                "step": step,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "best_val_loss": best_val_loss,
                "config": model_cfg.__dict__,
            }, latest_ckpt)

    print(f"Pre-training finished. Best Val Loss: {best_val_loss:.4f}")
    if wandb_cfg.enabled:
        wandb.finish()


if __name__ == "__main__":
    main()
