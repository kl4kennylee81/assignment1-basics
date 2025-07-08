import regex as re

# pretokenization
def pretokenize(text, special_tokens=None):
  PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
  if special_tokens is not None:
     special_pattern ='|'.join(re.escape(token) for token in special_tokens)
     PAT = f"({special_pattern})|({PAT})"
  output = dict()
  for subword in re.finditer(PAT, text):
    key = tuple(subword.group(0).encode("utf-8"))
    if key not in output:
      output[key] = 0
    output[key] += 1
  return output

def byte_pair_counts(pretoken_counts, special_tokens=[]):
  special_byte_tuples = set()
  for token in special_tokens:
    special_byte_tuples.add(tuple(token.encode("utf-8")))
  pair_counts = dict()
  pair_pretokens = dict()
  for key in pretoken_counts:
    if key in special_byte_tuples:
      continue
    if len(key) == 1:
      continue
    for i in range(len(key) - 1):
      pair = (key[i], key[i + 1])
      if pair not in pair_counts:
        pair_counts[pair] = 0
      pair_counts[pair] += pretoken_counts[key]
      if pair not in pair_pretokens:
        pair_pretokens[pair] = []
      pair_pretokens[pair].append(key)
  return pair_counts, pair_pretokens

def replace_byte_pair(pretokens, byte_pair, token):
    output = []
    i = 0
    while i < len(pretokens):
        if i < len(pretokens) - 1 and pretokens[i] == byte_pair[0] and pretokens[i+1] == byte_pair[1]:
            output.append(token)
            i += 2
        else:
            output.append(pretokens[i])
            i += 1
    return tuple(output)

class BPETokenizer:
    def __init__(self, input_path, vocab_size, special_tokens=None):
        assert(vocab_size > 256)
        self.input_path = input_path
        self.vocab_size = vocab_size
        self.merges = dict()
        self.output_merges = []
        self.vocab = dict()
        self.output_vocab = dict()
        self.special_tokens = []
        
        # Initialize with 256 byte tokens
        for i in range(256):
            self.vocab[i] = bytes([i])
        
        self.next_token = max(self.vocab.keys()) + 1  # 256
        
        # Add special tokens if provided
        if special_tokens != None:
            self.special_tokens = special_tokens
        for special_token in special_tokens:
            byte_encoded = special_token.encode("utf-8")
            self.vocab[self.next_token] = byte_encoded
            self.next_token += 1
        
        self.num_merges = max(0, vocab_size - len(self.vocab))

    def train(self):
        with open(self.input_path, "r", encoding="utf-8") as f:
            text = f.read()
        pretokens = pretokenize(text)
        
        for step in range(self.num_merges):
            pair_counts, pair_pretokens = byte_pair_counts(pretokens)
            
            # Handle case where no pairs exist
            if not pair_counts:
                break
                
            pair = max(pair_counts, key=lambda x: (pair_counts[x], x))
            self.merges[pair] = self.next_token
            self.vocab[self.next_token] = pair
            
            for pretoken in pair_pretokens[pair]:
                pretoken_count = pretokens.pop(pretoken, 0)
                new_pretoken = replace_byte_pair(pretoken, pair, self.next_token)
                pretokens[new_pretoken] = pretokens.get(new_pretoken, 0) + pretoken_count
            
            self.next_token += 1

        unmerge = {v: k for k, v in self.merges.items()}
        def unmerge_token(token):
            """Recursively unmerge a single token"""
            if token < 256:
                return bytes([token])  # Base byte token
            elif isinstance(self.vocab[token], bytes):
                return self.vocab[token]  # Special token
            elif token in unmerge:
                # Merged token - recursively get bytes from components
                pair = unmerge[token]
                return unmerge_token(pair[0]) + unmerge_token(pair[1])
        for token in self.vocab:
            self.output_vocab[token] = unmerge_token(token)

        for pair in self.merges:
          self.output_merges.append((self.output_vocab[pair[0]], self.output_vocab[pair[1]]))