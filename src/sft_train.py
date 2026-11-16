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
    parser.add_argument("--base_model", type=str, default="checkpoints/best_pretrained_42m.pt")
    parser.add_argument("--output_dir", type=str, default="checkpoints")
    parser.add_argument("--data_dir", type=str, default="data")
    parser.add_argument("--tokenizer_path", type=str, default=None)
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

    tok_path = args.tokenizer_path or os.path.join(args.data_dir, "tokenizer.json")
    tokenizer = get_tokenizer(tok_path if os.path.exists(tok_path) else None)
    model_cfg.vocab_size = tokenizer.vocab_size

    print(f"Starting Chat SFT (Phase 2) | Checkpoint: {sft_cfg.checkpoint_path}")

    # load pre-trained weights
    ckpt_path = sft_cfg.checkpoint_path
    if not os.path.exists(ckpt_path):
        fallback_path = ckpt_path.replace("42m", "28m") if "42m" in ckpt_path else ckpt_path.replace("28m", "42m")
        if os.path.exists(fallback_path):
            print(f"'{ckpt_path}' not found, falling back to '{fallback_path}'")
            ckpt_path = fallback_path

    if os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location=device)
        state_dict = ckpt.get("model_state_dict", ckpt)
        if "config" in ckpt and "n_layer" in ckpt["config"]:
            model_cfg.n_layer = ckpt["config"]["n_layer"]
        if "transformer.wte.weight" in state_dict:
            model_cfg.vocab_size = state_dict["transformer.wte.weight"].shape[0]
        model = SmallLLM(model_cfg).to(device)
        model.load_state_dict(state_dict, strict=False)
        print(f"Pre-trained weights loaded from '{ckpt_path}'.")
    else:
        print(f"[Warning] '{ckpt_path}' not found, training from random init.")
        model = SmallLLM(model_cfg).to(device)

    dataloader = load_sft_data(tokenizer=tokenizer, max_samples=10000, batch_size=sft_cfg.batch_size)
    data_iter = iter(dataloader)

    optimizer = torch.optim.AdamW(model.parameters(), lr=sft_cfg.learning_rate, betas=(0.9, 0.95), weight_decay=0.01)

    if wandb_cfg.enabled:
        wandb.init(
            project=wandb_cfg.project,
            name="tinystories-42m-chat-sft",
            tags=["sft", "chat", "42M", "alpaca"],
            config={**model_cfg.__dict__, **sft_cfg.__dict__}
        )
        chat_table = wandb.Table(columns=["Step", "Instruction Query", "Assistant Response"])

    test_queries = [
        "Write a 2-sentence story about a friendly mouse sharing an apple with a bird.",
        "Tell me what happened when Lily found a little bird that could not fly.",
        "Why is it important to be kind to our friends in the playground?",
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
                    repetition_penalty=1.2,
                    stop_id=[tokenizer.im_end_id, tokenizer.eos_id]
                )
                new_tokens = gen_tokens[0, p_tokens.size(1):].tolist()
                reply = tokenizer.extract_reply(tokenizer.decode(new_tokens))
                print(f"  Q: {q}\n  A: {reply}\n")
                if wandb_cfg.enabled:
                    chat_table.add_data(step, q, reply)

            if wandb_cfg.enabled:
                wandb.log({"samples/chat_table": chat_table})
            model.train()

    # save final chatbot weights
    os.makedirs(args.output_dir, exist_ok=True)
    final_path = os.path.join(args.output_dir, "chat_model_42m.pt")
    torch.save({
        "model_state_dict": {k: v.half() for k, v in model.state_dict().items()},
        "config": model_cfg.__dict__,
    }, final_path)
    print(f"Chatbot weights saved -> '{final_path}'")

    if wandb_cfg.enabled:
        art = wandb.Artifact("chat-42m-final", type="model")
        art.add_file(final_path)
        wandb.log_artifact(art, aliases=["latest", "production"])
        wandb.finish()


if __name__ == "__main__":
    main()
