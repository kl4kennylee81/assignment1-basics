import regex as re
import json
import base64
from typing import Iterable, Iterator

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
      pretokens = pretokenize_for_encoding(text, self.special_tokens)
      all_tokens = []
      
      for pretoken in pretokens:
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
              # Handle regular token - convert to individual bytes and apply merges
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