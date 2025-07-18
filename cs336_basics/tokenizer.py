# Standard library imports
import argparse
import base64
import json
import os
import sys
from pathlib import Path
from typing import Iterable, Iterator

# Third-party imports
import numpy as np
import regex as re
from tqdm import tqdm

def pretokenize_for_encoding(text, special_tokens=None):
    """Tokenize text into a list of byte tuples for encoding."""
    pattern = re.compile(r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")
    
    if not special_tokens:
        # Fast path for no special tokens
        matches = list(pattern.finditer(text))
        result = []
        for match in matches:
            token_bytes = tuple(match.group(0).encode("utf-8"))
            result.append(token_bytes)
        return result
    
    # Sort special tokens by length (longest first) to handle overlapping tokens correctly
    special_tokens_sorted = sorted(special_tokens, key=len, reverse=True)
    special_tokens_set = set(special_tokens)
    
    # Create pattern that matches longest tokens first
    special_pattern = '|'.join(re.escape(token) for token in special_tokens_sorted)
    
    # Split text on special tokens
    text_segments = re.split(f'({special_pattern})', text)
    
    result = []
    for segment in text_segments:
        if not segment:
            continue
        if segment in special_tokens_set:
            result.append(tuple(segment.encode("utf-8")))
        else:
            # Process regular text segments
            matches = list(pattern.finditer(segment))
            for match in matches:
                token_bytes = tuple(match.group(0).encode("utf-8"))
                result.append(token_bytes)
    
    return result

class Tokenizer:
    def __init__(self, vocab: dict[int, bytes], merges: list[tuple[bytes, bytes]], special_tokens: list[str] | None = None):
        self.vocab = vocab
        self._rvocab = {v: k for k, v in vocab.items()}
        self.merges = merges
        self.special_tokens = special_tokens
        # Create merge lookup table for O(1) access
        self.merge_lookup = {}
        for i, (left, right) in enumerate(merges):
            pair = (left, right)
            self.merge_lookup[pair] = i

    @classmethod
    def from_files(cls, vocab_filepath: str, merge_filepath:str, special_tokens: list[str] | None = None) -> "Tokenizer":
        with open(vocab_filepath, "r", encoding='utf-8') as f:
            vocab_data = json.load(f)
            vocab = {int(k): base64.b64decode(v.encode('utf-8')) for k, v in vocab_data.items()}       

        with open(merge_filepath, "r", encoding='utf-8') as f:
            merges = []
            for line in f:
                parts = line.rstrip().split(" ")
                if len(parts) == 2:
                    left = base64.b64decode(parts[0])
                    right = base64.b64decode(parts[1])
                    merges.append((left, right))
        return Tokenizer(vocab, merges, special_tokens)

    def encode(self, text: str) -> list[int]:
        # Get pretokens (each is a tuple of bytes)
        print("Pretokenizing...")
        pretokens = pretokenize_for_encoding(text, self.special_tokens)
        all_tokens = []
        
        # Add progress bar for processing pretokens
        for pretoken in tqdm(pretokens, desc="Encoding pretokens", unit="pretoken"):
            # Check if this pretoken is a special token
            pretoken_bytes = bytes(pretoken)
            pretoken_str = pretoken_bytes.decode('utf-8', errors='ignore')
            
            if self.special_tokens and pretoken_str in self.special_tokens:
                # Handle special token - look it up directly in vocab
                if pretoken_bytes in self._rvocab:
                    special_token_id = self._rvocab[pretoken_bytes]
                    all_tokens.append(special_token_id)
                else:
                    raise ValueError(f"Special token '{pretoken_str}' not found in vocabulary")
            else:
                # Handle regular token - convert to individual bytes first
                tokens = []
                for byte_val in pretoken:
                    single_byte = bytes([byte_val])  # Convert int to bytes object
                    if single_byte in self._rvocab:
                        token_id = self._rvocab[single_byte]
                        tokens.append(token_id)
                    else:
                        raise ValueError(f"Byte {single_byte} (ASCII {byte_val}) not found in vocabulary")
                
                # Apply merges using efficient algorithm
                while True:
                    # Find the earliest merge available
                    earliest_merge = None  # (position, merge_index)
                    
                    for i in range(len(tokens) - 1):
                        # Get the byte pair at position i
                        left_bytes = self.vocab[tokens[i]]
                        right_bytes = self.vocab[tokens[i + 1]]
                        pair = (left_bytes, right_bytes)
                        
                        # Check if this pair has a merge rule
                        merge_index = self.merge_lookup.get(pair, -1)
                        if merge_index != -1:
                            # If this is the earliest merge found so far, save it
                            if earliest_merge is None or merge_index < earliest_merge[1]:
                                earliest_merge = (i, merge_index)
                    
                    # If no merge found, we're done
                    if earliest_merge is None:
                        break
                    
                    # Apply the earliest merge
                    pos, merge_idx = earliest_merge
                    left_bytes = self.vocab[tokens[pos]]
                    right_bytes = self.vocab[tokens[pos + 1]]
                    merged_bytes = left_bytes + right_bytes
                    
                    if merged_bytes in self._rvocab:
                        merged_token_id = self._rvocab[merged_bytes]
                        # Replace the two tokens with the merged token
                        tokens = tokens[:pos] + [merged_token_id] + tokens[pos + 2:]
                    else:
                        # This shouldn't happen if vocab is consistent with merges
                        break
                
                # Add processed tokens from this pretoken to final result
                all_tokens.extend(tokens)
        
        return all_tokens

    def decode(self, ids: list[int]) -> str:
        decoded_bytes = b''.join(self.vocab[id] for id in ids)
        return decoded_bytes.decode("utf-8", errors="replace")

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        for chunk in iterable:
            # Use your existing pretokenize_for_encoding function
            pretokens = pretokenize_for_encoding(chunk, self.special_tokens)
            
            for pretoken in pretokens:
                # Check if this pretoken is a special token
                pretoken_bytes = bytes(pretoken)
                pretoken_str = pretoken_bytes.decode('utf-8', errors='ignore')
                
                if self.special_tokens and pretoken_str in self.special_tokens:
                    # Handle special token - look it up directly in vocab
                    if pretoken_bytes in self._rvocab:
                        special_token_id = self._rvocab[pretoken_bytes]
                        yield special_token_id
                    else:
                        raise ValueError(f"Special token '{pretoken_str}' not found in vocabulary")
                    continue
                
                # Handle regular token - convert to individual bytes first
                tokens = []
                for byte_val in pretoken:
                    single_byte = bytes([byte_val])  # Convert int to bytes object
                    if single_byte in self._rvocab:
                        token_id = self._rvocab[single_byte]
                        tokens.append(token_id)
                    else:
                        raise ValueError(f"Byte {single_byte} (ASCII {byte_val}) not found in vocabulary")
                
                # Apply merges to this pretoken
                for left_bytes, right_bytes in self.merges:
                    new_tokens = []
                    i = 0
                    while i < len(tokens):
                        # Check if we can merge at position i
                        if (i < len(tokens) - 1 and 
                            self.vocab[tokens[i]] == left_bytes and 
                            self.vocab[tokens[i + 1]] == right_bytes):
                            # Merge: find token ID for merged bytes
                            merged_bytes = left_bytes + right_bytes
                            if merged_bytes in self._rvocab:
                                merged_token_id = self._rvocab[merged_bytes]
                                new_tokens.append(merged_token_id)
                                i += 2  # Skip both tokens
                            else:
                                new_tokens.append(tokens[i])
                                i += 1
                        else:
                            new_tokens.append(tokens[i])
                            i += 1
                    tokens = new_tokens
                
                # Yield the processed tokens from this pretoken
                for token_id in tokens:
                    yield token_id

def roundtrip_tinystories_sample(file_path):
    """
    Encode and decode a text file using TinyStories tokenizer.
    
    Args:
        file_path: Path to the text file to encode
    """
    # Validate input file exists
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Input file not found: {file_path}")
    
    # Get original file size
    original_size = os.path.getsize(file_path)
    
    print("Loading tokenizer...")
    tokenizer = Tokenizer.from_files("output/tinystories_vocab.json", "output/tinystories_merges.txt", ["<endoftext>"])
    
    print("Reading file...")
    with open(file_path, "r", encoding="utf-8") as f:
        text = f.read()
    
    print("Encoding text...")
    encoded = tokenizer.encode(text)
    
    # Create output filename based on input filename
    input_path = Path(file_path)
    output_filename = f"{input_path.stem}_encoded.npy"
    output_path = Path("output") / output_filename
    
    # Ensure output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    print("Saving array...")
    encoded_array = np.array(encoded, dtype=np.uint16)
    np.save(output_path, encoded_array)
    encoded_size = os.path.getsize(output_path)
    
    print("Verifying round-trip...")
    loaded_tokens = np.load(output_path)
    token_ids = loaded_tokens.tolist()
    decoded_text = tokenizer.decode(token_ids)
    
    # Results summary
    print(f"\nResults:")
    print(f"File: {file_path}")
    print(f"Original: {original_size:,} bytes ({original_size / (1024*1024):.2f} MB)")
    print(f"Encoded: {encoded_size:,} bytes ({encoded_size / (1024*1024):.2f} MB)")
    print(f"Compression: {original_size / encoded_size:.2f}x")
    print(f"Tokens: {len(encoded):,}")
    print(f"Output: {output_path}")
    
    # Verify round-trip
    if decoded_text == text:
        print("✓ Round-trip successful")
    else:
        print("⚠ Round-trip failed")
        for i, (a, b) in enumerate(zip(text, decoded_text)):
            if a != b:
                print(f"First diff at pos {i}: '{a}' vs '{b}'")
                break
    
    # Verify round-trip
    if decoded_text == text:
        print("✓ Round-trip successful")
    else:
        print("⚠ Round-trip failed")
        for i, (a, b) in enumerate(zip(text, decoded_text)):
            if a != b:
                print(f"First diff at pos {i}: '{a}' vs '{b}'")
                break

def main():
    parser = argparse.ArgumentParser(description="Encode text file using TinyStories tokenizer")
    parser.add_argument("file_path", 
                       help="Path to the text file to encode")
    parser.add_argument("--output-dir", default="output",
                       help="Output directory for encoded file (default: output)")
    
    args = parser.parse_args()
    
    try:
        roundtrip_tinystories_sample(args.file_path)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        exit(1)
    except Exception as e:
        print(f"Error processing file: {e}")
        exit(1)

if __name__ == "__main__":
    main()

# Example usage:
# python cs336_basics/tokenizer.py data/TinyStoriesV2-GPT4-valid.txt
# python cs336_basics/tokenizer.py data/TinyStoriesV2-GPT4-train.txt