from typing import List, Dict, Optional


class ChatMLTokenizer:
    SPECIAL_TOKENS = {
        "<|endoftext|>": 50256,
        "<|im_start|>": 50257,
        "<|im_end|>": 50258,
        "<|pad|>": 50259,
    }

    def __init__(self, backend: str = "tiktoken"):
        self.backend = backend
        self.eos_id = self.SPECIAL_TOKENS["<|endoftext|>"]
        self.im_start_id = self.SPECIAL_TOKENS["<|im_start|>"]
        self.im_end_id = self.SPECIAL_TOKENS["<|im_end|>"]
        self.pad_id = self.SPECIAL_TOKENS["<|pad|>"]

        if backend == "tiktoken":
            try:
                import tiktoken
            except ImportError as err:
                raise ImportError("Run: pip install tiktoken") from err

            base_enc = tiktoken.get_encoding("gpt2")
            
            self.enc = tiktoken.Encoding(
                name="gpt2_chatml",
                pat_str=base_enc._pat_str,
                mergeable_ranks=base_enc._mergeable_ranks,
                special_tokens={
                    **base_enc._special_tokens,
                    **self.SPECIAL_TOKENS
                }
            )
            self.vocab_size = self.enc.n_vocab
        else:
            raise NotImplementedError(f"Backend '{backend}' not supported.")

    def encode(self, text: str, allowed_special: str = "all") -> List[int]:
        return self.enc.encode(text, allowed_special=allowed_special)

    def decode(self, tokens: List[int]) -> str:
        return self.enc.decode(tokens)

    def format_chatml(self, messages: List[Dict[str, str]]) -> str:
        out = ""
        for msg in messages:
            out += f"<|im_start|>{msg['role']}\n{msg['content'].strip()}<|im_end|>\n"
        return out

    def extract_reply(self, raw_output: str) -> str:
        if "<|im_start|>assistant\n" in raw_output:
            reply = raw_output.split("<|im_start|>assistant\n")[-1]
            return reply.replace("<|im_end|>", "").strip()
        return raw_output.replace("<|im_end|>", "").strip()


def get_tokenizer(backend: str = "tiktoken") -> ChatMLTokenizer:
    return ChatMLTokenizer(backend=backend)


if __name__ == "__main__":
    tok = get_tokenizer()
    sample = "Once upon a time in a small village"
    ids = tok.encode(sample)
    print(f"Vocab size: {tok.vocab_size} | Encoded: {ids} | Decoded: '{tok.decode(ids)}'")
