import os
import math
import argparse
from contextlib import nullcontext
import torch
import wandb

from src.config import ModelConfig, SFTConfig, WandbConfig
from src.tokenizer import get_tokenizer
from src.dataset import load_sft_data
from src.model import SmallLLM


def parse_args():
    parser = argparse.ArgumentParser(description="Chat SFT on Tiny-Alpaca")
    parser.add_argument("--base_model", type=str, default="checkpoints/best_pretrained_28m.pt")
    parser.add_argument("--output_dir", type=str, default="checkpoints")
    parser.add_argument("--max_steps", type=int, default=1500)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=3e-5)
    parser.add_argument("--no_wandb", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    model_cfg = ModelConfig()
    sft_cfg = SFTConfig()
    wandb_cfg = WandbConfig()

    sft_cfg.max_steps = args.max_steps
    sft_cfg.batch_size = args.batch_size
    sft_cfg.learning_rate = args.lr
    sft_cfg.checkpoint_path = args.base_model
    if args.no_wandb:
        wandb_cfg.enabled = False

    device = "cuda" if torch.cuda.is_available() else "cpu"
    device_type = 'cuda' if 'cuda' in device else 'cpu'
    ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=torch.float16)
    tokenizer = get_tokenizer()

    print("=" * 60)
    print("Chat SFT Engine (Phase 2)")
    print(f"Base Checkpoint: {sft_cfg.checkpoint_path}")
    print("=" * 60)

    model = SmallLLM(model_cfg).to(device)

    # load pre-trained weights
    if os.path.exists(sft_cfg.checkpoint_path):
        ckpt = torch.load(sft_cfg.checkpoint_path, map_location=device)
        state_dict = ckpt.get("model_state_dict", ckpt)
        model.load_state_dict(state_dict, strict=False)
        print("Pre-trained weights loaded.")
    else:
        print(f"[Warning] '{sft_cfg.checkpoint_path}' not found, training from random init.")

    dataloader = load_sft_data(tokenizer=tokenizer, max_samples=10000, batch_size=sft_cfg.batch_size)
    data_iter = iter(dataloader)

    optimizer = torch.optim.AdamW(model.parameters(), lr=sft_cfg.learning_rate, betas=(0.9, 0.95), weight_decay=0.01)

    if wandb_cfg.enabled:
        wandb.init(
            project=wandb_cfg.project,
            name="tinystories-28m-chat-sft",
            tags=["sft", "chat", "28M", "alpaca"],
            config={**model_cfg.__dict__, **sft_cfg.__dict__}
        )
        chat_table = wandb.Table(columns=["Step", "Instruction Query", "Assistant Response"])

    test_queries = [
        "What should you do if you see a lost dog in the park?",
        "Can you explain why the sky looks blue in simple words?",
        "Write a 2-sentence story about a friendly mouse sharing an apple.",
    ]

    model.train()
    for step in range(sft_cfg.max_steps):
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(dataloader)
            batch = next(data_iter)

        input_ids = batch["input_ids"].to(device)
        targets = batch["targets"].to(device)

        optimizer.zero_grad(set_to_none=True)
        with ctx:
            logits, loss = model(input_ids, targets=targets)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        if step % 25 == 0:
            ppl = math.exp(min(loss.item(), 20.0))
            print(f"SFT Step {step:4d}/{sft_cfg.max_steps} | Loss: {loss.item():.4f} | Perplexity: {ppl:.2f}")
            if wandb_cfg.enabled:
                wandb.log({"sft/step": step, "sft/loss": loss.item(), "sft/perplexity": ppl})

        # evaluate assistant generation
        if step % sft_cfg.eval_interval == 0 and step > 0:
            model.eval()
            print(f"\n[EVAL Step {step}]")
            for q in test_queries:
                prompt_str = f"<|im_start|>user\n{q}<|im_end|>\n<|im_start|>assistant\n"
                p_tokens = torch.tensor(tokenizer.encode(prompt_str), dtype=torch.long, device=device).unsqueeze(0)
                gen_tokens = model.generate(
                    p_tokens,
                    max_new_tokens=65,
                    temperature=0.7,
                    top_k=30,
                    top_p=0.9,
                    stop_id=tokenizer.im_end_id
                )
                raw_output = tokenizer.decode(gen_tokens[0].tolist())
                reply = tokenizer.extract_reply(raw_output)
                print(f"  Q: {q}\n  A: {reply}\n")
                if wandb_cfg.enabled:
                    chat_table.add_data(step, q, reply)

            if wandb_cfg.enabled:
                wandb.log({"samples/chat_table": chat_table})
            model.train()

    # save final chatbot weights
    os.makedirs(args.output_dir, exist_ok=True)
    final_path = os.path.join(args.output_dir, "chat_model_28m.pt")
    torch.save({
        "model_state_dict": {k: v.half() for k, v in model.state_dict().items()},
        "config": model_cfg.__dict__,
    }, final_path)
    print(f"Chatbot weights saved -> '{final_path}'")

    if wandb_cfg.enabled:
        art = wandb.Artifact("chat-28m-final", type="model")
        art.add_file(final_path)
        wandb.log_artifact(art, aliases=["latest", "production"])
        wandb.finish()


if __name__ == "__main__":
    main()
