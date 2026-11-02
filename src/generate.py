import os
import time
import argparse
import torch

from src.config import ModelConfig
from src.tokenizer import get_tokenizer
from src.model import SmallLLM


def parse_args():
    parser = argparse.ArgumentParser(description="Interactive Inference with Custom 28M LLM")
    parser.add_argument("--model_path", type=str, default="checkpoints/chat_model_28m.pt")
    parser.add_argument("--mode", type=str, choices=["chat", "story"], default="chat")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_k", type=int, default=35)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--max_tokens", type=int, default=80)
    return parser.parse_args()


def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = get_tokenizer()
    config = ModelConfig()

    print("=" * 60)
    print(f"28M LLM Terminal | Device: {device.upper()} | Mode: {args.mode.upper()}")
    print("=" * 60)

    if not os.path.exists(args.model_path):
        print(f"[Error] Checkpoint not found at: '{args.model_path}'")
        return

    model = SmallLLM(config).to(device)
    ckpt = torch.load(args.model_path, map_location=device)
    state_dict = ckpt.get("model_state_dict", ckpt)
    model.load_state_dict(state_dict)
    model.eval()

    if args.mode == "chat":
        print("Chat Mode (Type 'exit' or 'quit' to stop)\n")
        while True:
            try:
                user_msg = input("You > ").strip()
                if not user_msg:
                    continue
                if user_msg.lower() in ["exit", "quit", "q"]:
                    break

                prompt_str = f"<|im_start|>user\n{user_msg}<|im_end|>\n<|im_start|>assistant\n"
                input_tokens = torch.tensor(tokenizer.encode(prompt_str), dtype=torch.long, device=device).unsqueeze(0)

                t0 = time.time()
                out_tokens = model.generate(
                    input_tokens,
                    max_new_tokens=args.max_tokens,
                    temperature=args.temperature,
                    top_k=args.top_k,
                    top_p=args.top_p,
                    stop_id=tokenizer.im_end_id
                )
                dt = time.time() - t0

                raw_output = tokenizer.decode(out_tokens[0].tolist())
                reply = tokenizer.extract_reply(raw_output)
                tok_count = out_tokens.size(1) - input_tokens.size(1)

                print(f"Chatbot > {reply}")
                print(f"          [{tok_count} tokens in {dt:.2f}s ({tok_count / max(dt, 1e-4):.1f} tok/s)]\n")

            except (KeyboardInterrupt, EOFError):
                break

    elif args.mode == "story":
        print("Story Mode (Type 'exit' or 'quit' to stop)\n")
        while True:
            try:
                prompt_text = input("Prompt > ").strip()
                if not prompt_text:
                    continue
                if prompt_text.lower() in ["exit", "quit", "q"]:
                    break

                input_tokens = torch.tensor(tokenizer.encode(prompt_text), dtype=torch.long, device=device).unsqueeze(0)
                t0 = time.time()
                out_tokens = model.generate(
                    input_tokens,
                    max_new_tokens=args.max_tokens,
                    temperature=args.temperature,
                    top_k=args.top_k,
                    top_p=args.top_p
                )
                dt = time.time() - t0

                completion = tokenizer.decode(out_tokens[0].tolist())
                tok_count = out_tokens.size(1) - input_tokens.size(1)
                print(f"\n{completion}")
                print(f"[{tok_count} tokens in {dt:.2f}s ({tok_count / max(dt, 1e-4):.1f} tok/s)]\n")

            except (KeyboardInterrupt, EOFError):
                break


if __name__ == "__main__":
    main()
