import os
from typing import Tuple, Optional, List, Dict
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from src.tokenizer import ChatMLTokenizer, get_tokenizer


class PretrainingBuffer:
    """Memory-mapped binary buffer for streaming tokens to GPU with zero RAM overhead."""
    def __init__(self, data_path: str, block_size: int = 256):
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"Missing '{data_path}'. Run prepare_tinystories() first.")
        self.block_size = block_size
        self.data = np.memmap(data_path, dtype=np.uint16, mode='r')
        self.total_tokens = len(self.data)

    def get_batch(self, batch_size: int, device: str) -> Tuple[torch.Tensor, torch.Tensor]:
        ix = np.random.randint(0, self.total_tokens - self.block_size, (batch_size,))
        x_np = np.stack([self.data[i : i + self.block_size].astype(np.int64) for i in ix])
        y_np = np.stack([self.data[i + 1 : i + 1 + self.block_size].astype(np.int64) for i in ix])
        x = torch.from_numpy(x_np).to(device)
        y = torch.from_numpy(y_np).to(device)
        return x, y


class SFTDataset(Dataset):
    """ChatML formatted instruction dataset with user prompt masking (ignore_index = -100)."""
    def __init__(
        self,
        data: List[Dict[str, str]],
        tokenizer: ChatMLTokenizer,
        max_length: int = 256
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.examples = []

        for item in data:
            instruction = item.get("instruction", "").strip()
            user_input = item.get("input", "").strip()
            response = item.get("output", "").strip()
            if not instruction or not response:
                continue

            full_prompt = f"{instruction}\n\nContext: {user_input}" if user_input else instruction
            prompt_str = f"<|im_start|>user\n{full_prompt}<|im_end|>\n<|im_start|>assistant\n"
            response_str = f"{response}<|im_end|>"

            prompt_tokens = self.tokenizer.encode(prompt_str)
            response_tokens = self.tokenizer.encode(response_str)

            input_ids = prompt_tokens + response_tokens
            # mask prompt: loss is only calculated on assistant tokens
            targets = [-100] * len(prompt_tokens) + response_tokens

            if len(input_ids) > max_length:
                input_ids = input_ids[:max_length]
                targets = targets[:max_length]

            pad_len = max_length - len(input_ids)
            if pad_len > 0:
                input_ids = input_ids + [self.tokenizer.pad_id] * pad_len
                targets = targets + [-100] * pad_len

            self.examples.append({
                "input_ids": torch.tensor(input_ids, dtype=torch.long),
                "targets": torch.tensor(targets, dtype=torch.long)
            })

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return self.examples[idx]


def prepare_tinystories(
    output_dir: str = "data",
    val_ratio: float = 0.05,
    max_samples: Optional[int] = None
) -> Tuple[str, str]:
    """Downloads and converts TinyStories into binary train.bin and val.bin files."""
    from datasets import load_dataset
    from tqdm import tqdm

    os.makedirs(output_dir, exist_ok=True)
    train_bin_path = os.path.join(output_dir, "train.bin")
    val_bin_path = os.path.join(output_dir, "val.bin")

    if os.path.exists(train_bin_path) and os.path.exists(val_bin_path):
        print(f"Tokenized files found in '{output_dir}', skipping preparation.")
        return train_bin_path, val_bin_path

    print("Fetching 'roneneldan/TinyStories' from Hugging Face...")
    ds = load_dataset("roneneldan/TinyStories", split="train")
    if max_samples:
        ds = ds.select(range(min(len(ds), max_samples)))

    tokenizer = get_tokenizer()
    total_len = len(ds)
    val_size = max(int(total_len * val_ratio), 1000)
    train_size = total_len - val_size

    def tokenize_and_save(split, save_path: str, desc: str):
        all_tokens = []
        for row in tqdm(split, desc=desc):
            text = row.get("text", "").strip()
            if text:
                tokens = tokenizer.encode(text) + [tokenizer.eos_id]
                all_tokens.extend(tokens)
        arr = np.array(all_tokens, dtype=np.uint16)
        arr.tofile(save_path)
        print(f"Saved {len(arr):,} tokens -> '{save_path}' ({os.path.getsize(save_path) / 1e6:.1f} MB)")
        return save_path

    tokenize_and_save(ds.select(range(train_size)), train_bin_path, "Tokenizing Train")
    tokenize_and_save(ds.select(range(train_size, total_len)), val_bin_path, "Tokenizing Val")
    return train_bin_path, val_bin_path


def load_sft_data(tokenizer: ChatMLTokenizer, max_samples: int = 10000, batch_size: int = 16) -> DataLoader:
    from datasets import load_dataset
    print("Loading instruction dataset (alpaca)...")
    ds = load_dataset("tatsu-lab/alpaca", split="train")
    if max_samples:
        ds = ds.select(range(min(len(ds), max_samples)))

    raw_data = [
        {"instruction": r["instruction"], "input": r.get("input", ""), "output": r["output"]}
        for r in ds
    ]
    sft_ds = SFTDataset(raw_data, tokenizer=tokenizer)
    return DataLoader(sft_ds, batch_size=batch_size, shuffle=True)
