import regex as re
from collections import defaultdict, Counter

def pretokenize(text, special_tokens=None):
    """Tokenize text into byte tuples, splitting on special tokens first."""
    pattern = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    
    if not special_tokens:
        # No special tokens - process entire text
        return Counter(tuple(match.group(0).encode("utf-8")) 
                      for match in re.finditer(pattern, text))
    
    # Split text on special tokens first
    special_pattern = '|'.join(re.escape(token) for token in special_tokens)
    text_segments = re.split(f'({special_pattern})', text)
    
    result = Counter()
    
    for segment in text_segments:
        if not segment:  # Skip empty segments
            continue
            
        if segment in special_tokens:
            # This is a special token - add it as-is
            result[tuple(segment.encode("utf-8"))] += 1
        else:
            # This is regular text - pretokenize it
            segment_tokens = Counter(tuple(match.group(0).encode("utf-8")) 
                                   for match in re.finditer(pattern, segment))
            result.update(segment_tokens)
    
    return result

def get_pair_counts(pretoken_counts, special_tokens=None):
    """Count all adjacent byte pairs across pretokens, skipping special tokens."""
    special_tokens = special_tokens or []
    
    # Convert special tokens to byte tuples for comparison
    special_byte_tuples = {tuple(token.encode("utf-8")) for token in special_tokens}
    
    pair_counts = defaultdict(int)
    pair_locations = defaultdict(list)
    
    for pretoken, count in pretoken_counts.items():
        # Skip special tokens - they should never have their bytes paired
        if pretoken in special_byte_tuples:
            continue
            
        if len(pretoken) < 2:
            continue
            
        for i in range(len(pretoken) - 1):
            pair = (pretoken[i], pretoken[i + 1])
            pair_counts[pair] += count
            pair_locations[pair].append(pretoken)
    
    return dict(pair_counts), dict(pair_locations)

def apply_merge(pretokens, old_pair, new_token, special_tokens=None):
    """Replace all occurrences of old_pair with new_token in pretokens."""
    special_tokens = special_tokens or []
    
    # Convert special tokens to byte tuples for comparison
    special_byte_tuples = {tuple(token.encode("utf-8")) for token in special_tokens}
    
    updated = {}
    
    for pretoken, count in pretokens.items():
        # Skip special tokens - they should never be modified
        if pretoken in special_byte_tuples:
            updated[pretoken] = count
            continue
            
        if len(pretoken) < 2 or old_pair[0] not in pretoken:
            updated[pretoken] = count
            continue
            
        # Build new pretoken by replacing pairs
        new_pretoken = []
        i = 0
        while i < len(pretoken):
            if (i < len(pretoken) - 1 and 
                pretoken[i] == old_pair[0] and 
                pretoken[i + 1] == old_pair[1]):
                new_pretoken.append(new_token)
                i += 2
            else:
                new_pretoken.append(pretoken[i])
                i += 1
        
        new_key = tuple(new_pretoken)
        updated[new_key] = updated.get(new_key, 0) + count
    
    return updated

class BPETokenizer:
    def __init__(self, input_path, vocab_size, special_tokens=None):
        assert vocab_size > 256, "Vocabulary size must be greater than 256"
        
        self.input_path = input_path
        self.vocab_size = vocab_size
        self.special_tokens = special_tokens or []
        
        # Initialize vocabulary: bytes + special tokens
        self.vocab = {i: bytes([i]) for i in range(256)}
        self.merges = {}
        
        # Add special tokens
        next_id = 256
        for token in self.special_tokens:
            self.vocab[next_id] = token.encode("utf-8")
            next_id += 1
        
        self.next_token_id = next_id
        self.max_merges = max(0, vocab_size - len(self.vocab))

    def _get_token_bytes(self, token_id):
        """Recursively resolve token to its byte representation."""
        if token_id < 256:
            return bytes([token_id])
        
        token_value = self.vocab[token_id]
        if isinstance(token_value, bytes):
            return token_value
        
        # It's a merge pair - recursively resolve both parts
        left_id, right_id = token_value
        return self._get_token_bytes(left_id) + self._get_token_bytes(right_id)

    def _select_best_pair(self, pair_counts):
        """Select the most frequent pair, using byte values for tie-breaking."""
        return max(pair_counts.keys(), key=lambda pair: (
            pair_counts[pair],
            self._get_token_bytes(pair[0]),
            self._get_token_bytes(pair[1])
        ))

    def train(self):
        """Train the BPE tokenizer on the input corpus."""
        # Load and preprocess text
        with open(self.input_path, "r", encoding="utf-8") as f:
            text = f.read()
        
        pretokens = pretokenize(text, self.special_tokens)
        
        # Perform BPE merges
        for _ in range(self.max_merges):
            pair_counts, pair_locations = get_pair_counts(pretokens, self.special_tokens)
            
            if not pair_counts:
                break
            
            # Select best pair and create new token
            best_pair = self._select_best_pair(pair_counts)
            self.vocab[self.next_token_id] = best_pair
            self.merges[best_pair] = self.next_token_id
            
            # Apply the merge
            pretokens = apply_merge(pretokens, best_pair, self.next_token_id, self.special_tokens)
            self.next_token_id += 1
        
        # Generate outputs
        self._build_outputs()

    def _build_outputs(self):
        """Build final vocabulary and merge list for external use."""
        self.output_vocab = {
            token_id: self._get_token_bytes(token_id) 
            for token_id in self.vocab
        }
        
        self.output_merges = [
            (self.output_vocab[pair[0]], self.output_vocab[pair[1]])
            for pair in self.merges
        ]

    def get_vocab_and_merges(self):
        """Return the trained vocabulary and merges."""
        return self.output_vocab, self.output_merges