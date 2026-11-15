import os
from typing import List, Dict, Optional, Iterable


def train_custom_bpe(
    texts: Iterable[str],
    save_path: str = "data/tokenizer.json",
    vocab_size: int = 8192,
    special_tokens: Optional[List[str]] = None,
) -> str:
    """Train a byte-level BPE tokenizer on text corpus and save to JSON."""
    try:
        from tokenizers import ByteLevelBPETokenizer
    except ImportError as err:
        raise ImportError("Run: pip install tokenizers") from err

    if special_tokens is None:
        special_tokens = ["<|endoftext|>", "<|im_start|>", "<|im_end|>", "<|pad|>"]

    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
    tokenizer = ByteLevelBPETokenizer()
    tokenizer.train_from_iterator(
        texts,
        vocab_size=vocab_size,
        min_frequency=2,
        special_tokens=special_tokens
    )
    tokenizer.save(save_path)
    return save_path


class ChatMLTokenizer:
    DEFAULT_SPECIAL_TOKENS = {
        "<|endoftext|>": 50256,
        "<|im_start|>": 50257,
        "<|im_end|>": 50258,
        "<|pad|>": 50259,
    }

    def __init__(
        self,
        tokenizer_path: Optional[str] = None,
        backend: Optional[str] = None
    ):
        self.tokenizer_path = tokenizer_path

        if tokenizer_path is not None and os.path.exists(tokenizer_path):
            self.backend = "custom"
        elif backend:
            self.backend = backend
        else:
            self.backend = "tiktoken"

        if self.backend == "custom":
            try:
                from tokenizers import Tokenizer
            except ImportError as err:
                raise ImportError("Run: pip install tokenizers") from err

            self.enc = Tokenizer.from_file(self.tokenizer_path)
            self.vocab_size = self.enc.get_vocab_size()

            self.eos_id = self.enc.token_to_id("<|endoftext|>")
            self.im_start_id = self.enc.token_to_id("<|im_start|>")
            self.im_end_id = self.enc.token_to_id("<|im_end|>")
            self.pad_id = self.enc.token_to_id("<|pad|>")

            if self.eos_id is None:
                self.eos_id = 0
            if self.im_start_id is None:
                self.im_start_id = 1
            if self.im_end_id is None:
                self.im_end_id = 2
            if self.pad_id is None:
                self.pad_id = 3

        elif self.backend == "tiktoken":
            try:
                import tiktoken
            except ImportError as err:
                raise ImportError("Run: pip install tiktoken") from err

            self.eos_id = self.DEFAULT_SPECIAL_TOKENS["<|endoftext|>"]
            self.im_start_id = self.DEFAULT_SPECIAL_TOKENS["<|im_start|>"]
            self.im_end_id = self.DEFAULT_SPECIAL_TOKENS["<|im_end|>"]
            self.pad_id = self.DEFAULT_SPECIAL_TOKENS["<|pad|>"]

            base_enc = tiktoken.get_encoding("gpt2")
            self.enc = tiktoken.Encoding(
                name="gpt2_chatml",
                pat_str=base_enc._pat_str,
                mergeable_ranks=base_enc._mergeable_ranks,
                special_tokens={
                    **base_enc._special_tokens,
                    **self.DEFAULT_SPECIAL_TOKENS
                }
            )
            self.vocab_size = self.enc.n_vocab
        else:
            raise NotImplementedError(f"Backend '{self.backend}' not supported.")

    def encode(self, text: str, allowed_special: str = "all") -> List[int]:
        if self.backend == "custom":
            return self.enc.encode(text).ids
        return self.enc.encode(text, allowed_special=allowed_special)

    def decode(self, tokens: List[int], skip_special_tokens: bool = False) -> str:
        if self.backend == "custom":
            return self.enc.decode(tokens, skip_special_tokens=skip_special_tokens)
        return self.enc.decode(tokens)

    def format_chatml(self, messages: List[Dict[str, str]]) -> str:
        out = ""
        for msg in messages:
            out += f"<|im_start|>{msg['role']}\n{msg['content'].strip()}<|im_end|>\n"
        return out

    def extract_reply(self, raw_output: str) -> str:
        if "<|im_start|>assistant\n" in raw_output:
            raw_output = raw_output.split("<|im_start|>assistant\n", 1)[1]

        for stop_marker in ["<|im_end|>", "<|im_start|>", "<|endoftext|>"]:
            if stop_marker in raw_output:
                raw_output = raw_output.split(stop_marker)[0]

        return raw_output.strip()


def get_tokenizer(
    tokenizer_path: Optional[str] = None,
    backend: Optional[str] = None
) -> ChatMLTokenizer:
    if backend == "tiktoken":
        return ChatMLTokenizer(backend="tiktoken")
    if tokenizer_path is not None and os.path.exists(tokenizer_path):
        return ChatMLTokenizer(tokenizer_path=tokenizer_path, backend="custom")
    default_path = os.path.join("data", "tokenizer.json")
    if tokenizer_path is None and os.path.exists(default_path):
        return ChatMLTokenizer(tokenizer_path=default_path, backend="custom")
    return ChatMLTokenizer(backend=backend or "tiktoken")


if __name__ == "__main__":
    tok = get_tokenizer()
    sample = "Once upon a time in a small village"
    ids = tok.encode(sample)
    print(f"Backend: {tok.backend} | Vocab size: {tok.vocab_size} | Encoded: {ids} | Decoded: '{tok.decode(ids)}'")
