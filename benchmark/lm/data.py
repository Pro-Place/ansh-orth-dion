"""
WikiText-103 data loading for language model experiments.

Uses HuggingFace datasets + GPT-2 tokenizer.
Packs tokens into fixed-length sequences for efficient training.
"""

import os
from typing import Optional, Tuple

import torch
from torch.utils.data import Dataset, DataLoader


class PackedTokenDataset(Dataset):
    """
    Pre-tokenized dataset that packs tokens into fixed-length sequences.
    Stores all tokens as a single flat tensor and slices windows.
    """

    def __init__(self, tokens: torch.Tensor, seq_len: int):
        """
        Args:
            tokens: 1D tensor of token ids
            seq_len: sequence length (context window)
        """
        self.tokens = tokens
        self.seq_len = seq_len
        # Number of complete sequences (need seq_len + 1 for input/target shift)
        self.n_sequences = (len(tokens) - 1) // seq_len

    def __len__(self) -> int:
        return self.n_sequences

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        start = idx * self.seq_len
        end = start + self.seq_len + 1
        chunk = self.tokens[start:end].long()
        x = chunk[:-1]   # input tokens
        y = chunk[1:]     # target tokens (shifted by 1)
        return x, y


def load_wikitext103(
    seq_len: int = 1024,
    cache_dir: Optional[str] = None,
    num_workers: int = 0,
) -> Tuple[PackedTokenDataset, PackedTokenDataset]:
    """
    Load and tokenize WikiText-103.

    Returns:
        (train_dataset, val_dataset)
    """
    try:
        from datasets import load_dataset
        from transformers import GPT2TokenizerFast
    except ImportError:
        raise ImportError(
            "Install dependencies: pip install datasets transformers"
        )

    if cache_dir is None:
        cache_dir = os.path.join(os.path.dirname(__file__), "..", "data", "wikitext103")

    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")

    # Check for cached tokenized tensors
    train_cache = os.path.join(cache_dir, "train_tokens.pt")
    val_cache = os.path.join(cache_dir, "val_tokens.pt")

    if os.path.exists(train_cache) and os.path.exists(val_cache):
        print(f"Loading cached tokens from {cache_dir}")
        train_tokens = torch.load(train_cache, weights_only=True)
        val_tokens = torch.load(val_cache, weights_only=True)
    else:
        print("Downloading and tokenizing WikiText-103...")
        dataset = load_dataset("wikitext", "wikitext-103-raw-v1", cache_dir=cache_dir)

        def tokenize_split(split_name: str) -> torch.Tensor:
            texts = dataset[split_name]["text"]
            # Filter empty lines and join
            text = "\n".join(t for t in texts if t.strip())
            tokens = tokenizer.encode(text)
            return torch.tensor(tokens, dtype=torch.int32)

        train_tokens = tokenize_split("train")
        val_tokens = tokenize_split("validation")

        os.makedirs(cache_dir, exist_ok=True)
        torch.save(train_tokens, train_cache)
        torch.save(val_tokens, val_cache)
        print(f"Cached tokens to {cache_dir}")

    print(f"Train: {len(train_tokens):,} tokens → "
          f"{(len(train_tokens)-1)//seq_len:,} sequences of length {seq_len}")
    print(f"Val:   {len(val_tokens):,} tokens → "
          f"{(len(val_tokens)-1)//seq_len:,} sequences of length {seq_len}")

    train_ds = PackedTokenDataset(train_tokens, seq_len)
    val_ds = PackedTokenDataset(val_tokens, seq_len)
    return train_ds, val_ds


def create_dataloaders(
    train_ds: PackedTokenDataset,
    val_ds: PackedTokenDataset,
    batch_size: int = 8,
    num_workers: int = 2,
) -> Tuple[DataLoader, DataLoader]:
    """Create train and validation DataLoaders."""
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )
    return train_loader, val_loader
