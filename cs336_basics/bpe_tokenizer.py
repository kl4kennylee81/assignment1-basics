import regex as re
from collections import defaultdict, Counter
import multiprocessing as mp
import os
from typing import BinaryIO
import heapq

def find_chunk_boundaries(file: BinaryIO, desired_num_chunks: int, split_special_token: bytes) -> list[int]:
    """Chunk the file into parts that can be counted independently."""
    assert isinstance(split_special_token, bytes), "Must represent special token as a bytestring"

    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)

    chunk_size = file_size // desired_num_chunks
    chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
    chunk_boundaries[-1] = file_size

    mini_chunk_size = 4096

    for bi in range(1, len(chunk_boundaries) - 1):
        initial_position = chunk_boundaries[bi]
        file.seek(initial_position)
        while True:
            mini_chunk = file.read(mini_chunk_size)
            if mini_chunk == b"":
                chunk_boundaries[bi] = file_size
                break
            found_at = mini_chunk.find(split_special_token)
            if found_at != -1:
                chunk_boundaries[bi] = initial_position + found_at
                break
            initial_position += mini_chunk_size

    return sorted(set(chunk_boundaries))

# Compile regex pattern at module level for better performance
_COMPILED_PATTERN = re.compile(r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")

def pretokenize(text, special_tokens=None):
    """Tokenize text into byte tuples, splitting on special tokens first."""
    if not special_tokens:
        # Fast path for no special tokens
        return Counter(tuple(match.group(0).encode("utf-8")) 
                      for match in _COMPILED_PATTERN.finditer(text))
    
    # Convert special tokens to set for faster lookup
    special_tokens_set = set(special_tokens)
    special_pattern = '|'.join(re.escape(token) for token in special_tokens)
    text_segments = re.split(f'({special_pattern})', text)
    
    result = Counter()
    for segment in text_segments:
        if not segment:
            continue
        if segment in special_tokens_set:
            result[tuple(segment.encode("utf-8"))] += 1
        else:
            # Batch process segment tokens
            segment_tokens = Counter(tuple(match.group(0).encode("utf-8")) 
                                   for match in _COMPILED_PATTERN.finditer(segment))
            result.update(segment_tokens)
    
    return result

def process_chunk(args):
    """Process a single chunk of the file - designed for multiprocessing."""
    filepath, start, end, special_tokens = args
    with open(filepath, "rb") as f:
        f.seek(start)
        chunk_bytes = f.read(end - start)
        chunk_text = chunk_bytes.decode("utf-8", errors="ignore")
    return pretokenize(chunk_text, special_tokens)

def parallel_pretokenize(filepath, special_tokens):
    num_processes = mp.cpu_count()
    with open(filepath, "rb") as f:
        boundaries = find_chunk_boundaries(f, num_processes, "<|endoftext|>".encode("utf-8"))
    
    chunk_args = [(filepath, start, end, special_tokens) for start, end in zip(boundaries[:-1], boundaries[1:])]
    with mp.Pool(num_processes) as pool:
        chunk_results = pool.map(process_chunk, chunk_args)
    
    combined_result = Counter()
    for result in chunk_results:
        combined_result.update(result)
    return combined_result

class OptimizedBPETrainer:
    """Optimized BPE trainer with incremental pair count updates."""
    
    def __init__(self, vocab_size, special_tokens=None):
        self.vocab_size = vocab_size
        self.special_tokens = special_tokens or []
        self.special_byte_tuples = {tuple(token.encode("utf-8")) for token in self.special_tokens}
        
        # Core data structures
        self.vocab = {i: bytes([i]) for i in range(256)}
        self.merges = {}
        
        # Add special tokens
        next_id = 256
        for token in self.special_tokens:
            self.vocab[next_id] = token.encode("utf-8")
            next_id += 1
        
        self.next_token_id = next_id
        self.max_merges = max(0, vocab_size - len(self.vocab))
        
        # Optimization: cached data structures
        self.pair_counts = defaultdict(int)
        self.pair_locations = defaultdict(set)  # Use set for faster removal
        self.pretoken_counts = {}

    def _get_token_bytes(self, token_id):
        """Recursively resolve token to its byte representation."""
        if token_id < 256:
            return bytes([token_id])
        
        token_value = self.vocab[token_id]
        if isinstance(token_value, bytes):
            return token_value
        
        left_id, right_id = token_value
        return self._get_token_bytes(left_id) + self._get_token_bytes(right_id)

    def _initialize_pair_counts(self):
        """Initialize pair counts from pretokens - only called once."""
        self.pair_counts.clear()
        self.pair_locations.clear()
        
        for pretoken, count in self.pretoken_counts.items():
            if pretoken in self.special_byte_tuples or len(pretoken) < 2:
                continue
            
            # Process all pairs in this pretoken at once
            pretoken_len = len(pretoken)
            for i in range(pretoken_len - 1):
                pair = (pretoken[i], pretoken[i + 1])
                self.pair_counts[pair] += count
                self.pair_locations[pair].add(pretoken)

    def _get_pairs_in_pretoken(self, pretoken):
        """Get all adjacent pairs in a pretoken."""
        pretoken_len = len(pretoken)
        if pretoken_len < 2:
            return []
        return [(pretoken[i], pretoken[i + 1]) for i in range(pretoken_len - 1)]

    def _apply_merge_optimized(self, old_pair, new_token):
        """Apply merge and incrementally update pair counts."""
        affected_pretokens = self.pair_locations[old_pair]
        
        # Process all affected pretokens
        pretokens_to_update = []
        for old_pretoken in affected_pretokens:
            if old_pretoken in self.pretoken_counts:
                pretokens_to_update.append((old_pretoken, self.pretoken_counts[old_pretoken]))
        
        for old_pretoken, count in pretokens_to_update:
            # Remove old pairs from counts
            old_pairs = self._get_pairs_in_pretoken(old_pretoken)
            for pair in old_pairs:
                self.pair_counts[pair] -= count
                self.pair_locations[pair].discard(old_pretoken)
                if self.pair_counts[pair] <= 0:
                    # Use pop() to avoid KeyError if key doesn't exist
                    self.pair_counts.pop(pair, None)
                    self.pair_locations.pop(pair, None)
            
            # Create new pretoken by applying merge
            new_pretoken = self._replace_pair_in_pretoken(old_pretoken, old_pair, new_token)
            
            # Update pretoken counts
            del self.pretoken_counts[old_pretoken]
            if new_pretoken in self.pretoken_counts:
                self.pretoken_counts[new_pretoken] += count
            else:
                self.pretoken_counts[new_pretoken] = count
            
            # Add new pairs to counts
            if new_pretoken not in self.special_byte_tuples:
                new_pairs = self._get_pairs_in_pretoken(new_pretoken)
                for pair in new_pairs:
                    self.pair_counts[pair] += count
                    self.pair_locations[pair].add(new_pretoken)

    def _replace_pair_in_pretoken(self, pretoken, old_pair, new_token):
        """Replace old_pair with new_token in pretoken."""
        result = []
        i = 0
        pretoken_len = len(pretoken)
        old_left, old_right = old_pair
        
        while i < pretoken_len:
            if (i < pretoken_len - 1 and 
                pretoken[i] == old_left and 
                pretoken[i + 1] == old_right):
                result.append(new_token)
                i += 2
            else:
                result.append(pretoken[i])
                i += 1
        return tuple(result)

    def _select_best_pair(self):
        """Select the most frequent pair with lexicographic tie-breaking."""
        if not self.pair_counts:
            return None
        
        # Cache the max count to avoid multiple iterations
        max_count = max(self.pair_counts.values())
        
        # Find best pair among those with max count
        best_pair = None
        best_key = None
        
        for pair, count in self.pair_counts.items():
            if count == max_count:
                # Create tie-breaking key
                key = (self._get_token_bytes(pair[0]), self._get_token_bytes(pair[1]))
                if best_key is None or key > best_key:
                    best_pair = pair
                    best_key = key
        
        return best_pair

    def train(self, pretokens):
        """Train BPE with optimized incremental updates."""
        self.pretoken_counts = dict(pretokens)
        
        # Initialize pair counts once
        self._initialize_pair_counts()
        # Perform merges with incremental updates
        for step in range(self.max_merges):
            best_pair = self._select_best_pair()
            if not best_pair:
                break
            
            # Record the merge
            self.vocab[self.next_token_id] = best_pair
            self.merges[best_pair] = self.next_token_id
            
            # Apply merge with incremental updates
            self._apply_merge_optimized(best_pair, self.next_token_id)
            self.next_token_id += 1

    def get_vocab_and_merges(self):
        """Build and return final vocabulary and merges."""
        output_vocab = {token_id: self._get_token_bytes(token_id) for token_id in self.vocab}
        output_merges = [(output_vocab[pair[0]], output_vocab[pair[1]]) for pair in self.merges]
        return output_vocab, output_merges

class BPETokenizer:
    """Simplified BPE tokenizer interface with optimized training."""
    
    def __init__(self, input_path, vocab_size, special_tokens=None):
        assert vocab_size > 256, "Vocabulary size must be greater than 256"
        self.input_path = input_path
        self.vocab_size = vocab_size
        self.special_tokens = special_tokens or []
        self.trainer = OptimizedBPETrainer(vocab_size, special_tokens)

    def train(self):
        """Train the BPE tokenizer with optimized merging."""
        
        # Skip multiprocessing for small files to avoid overhead
        file_size = os.path.getsize(self.input_path)
        if file_size > 50_000_000:  # 50MB threshold (increased from 10MB)
            pretokens = parallel_pretokenize(self.input_path, self.special_tokens)
        else:
            with open(self.input_path, "r", encoding="utf-8") as f:
                text = f.read()
            pretokens = pretokenize(text, self.special_tokens)
        
        # Train with optimized algorithm
        self.trainer.train(pretokens)
        
        # Get results
        self.output_vocab, self.output_merges = self.trainer.get_vocab_and_merges()

    def get_vocab_and_merges(self):
        """Return the trained vocabulary and merges."""
        return self.output_vocab, self.output_merges