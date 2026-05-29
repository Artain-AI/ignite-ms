"""Build a vocabulary cache from a tokenizer's vocabulary.

Pre-tokenizes all words from the tokenizer's vocab + common English words,
producing vocab_cache.bin for fast lookup during inference.

No corpus data needed — derived entirely from the tokenizer itself.

Requires: python3, tokenizers

Usage:
    python3 build_vocab_cache.py --tokenizer tokenizer.json --output vocab_cache.bin
"""

import argparse
import struct
import sys
import os

from tokenizers import Tokenizer


CACHE_MAGIC = b"IMSVCACH"
CACHE_VERSION = 1


def build_vocab_cache(tokenizer_path: str, output_path: str) -> None:
    tok = Tokenizer.from_file(tokenizer_path)
    vocab = tok.get_vocab()
    cache = {}

    print(f"[vocab_cache] tokenizer vocab size: {len(vocab)}", file=sys.stderr)

    # Extract words from the tokenizer vocabulary.
    # SentencePiece/BPE tokenizers prefix word-initial tokens with ▁ (▁).
    # We want whole-word entries: for each ▁xxx token, pre-tokenize "▁xxx".
    word_tokens = []
    for token in vocab:
        if token.startswith("▁") and len(token) > 1:
            word = token[1:]  # strip the ▁
            if len(word) <= 45 and any(c.isalpha() for c in word):
                word_tokens.append(token)

    print(f"[vocab_cache] word tokens to cache: {len(word_tokens)}", file=sys.stderr)

    # Pre-tokenize each word token
    for i, key in enumerate(word_tokens):
        enc = tok.encode(key, add_special_tokens=False)
        if enc.ids:
            cache[key] = enc.ids
        if (i + 1) % 100000 == 0:
            print(f"[vocab_cache]   progress: {i+1}/{len(word_tokens)}", file=sys.stderr)

    # Also add common subword fragments that appear frequently
    # (tokens without ▁ prefix that are still useful to cache)
    for token, token_id in vocab.items():
        if not token.startswith("▁") and len(token) >= 2 and len(token) <= 20:
            if any(c.isalpha() for c in token) and token not in cache:
                enc = tok.encode(token, add_special_tokens=False)
                if enc.ids:
                    cache[token] = enc.ids

    # Add normalized placeholder tokens used by ignite-ms normalizer
    for ph in ["<url>", "<email>", "<hash>", "<hex>", "<num>",
               "<version>", "<ref>", "<user>", "<sub>", "<path>"]:
        key = "▁" + ph
        enc = tok.encode(key, add_special_tokens=False)
        if enc.ids:
            cache[key] = enc.ids

    print(f"[vocab_cache] total cache entries: {len(cache)}", file=sys.stderr)

    # Write binary format
    with open(output_path, "wb") as f:
        f.write(CACHE_MAGIC)
        f.write(struct.pack("<I", CACHE_VERSION))
        f.write(struct.pack("<Q", len(cache)))
        for key, ids in cache.items():
            key_bytes = key.encode("utf-8")
            f.write(struct.pack("<I", len(key_bytes)))
            f.write(key_bytes)
            f.write(struct.pack("<I", len(ids)))
            for token_id in ids:
                f.write(struct.pack("<I", token_id))

    size_mb = os.path.getsize(output_path) / 1e6
    print(f"[vocab_cache] saved {output_path} ({size_mb:.1f} MB, {len(cache)} entries)", file=sys.stderr)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    build_vocab_cache(args.tokenizer, args.output)
