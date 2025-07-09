import regex as re
from collections import defaultdict, Counter
import multiprocessing as mp
import os
from typing import BinaryIO
import heapq
import time
from tqdm import tqdm

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

    print("Finding chunk boundaries...")
    for bi in tqdm(range(1, len(chunk_boundaries) - 1), desc="Processing boundaries"):
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

def get_pattern():
    """Get compiled regex pattern - safer for multiprocessing."""
    return re.compile(r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")


def pretokenize(text, special_tokens=None):
    """Tokenize text into byte tuples, splitting on special tokens first."""
    start_time = time.time()
    
    if not special_tokens:
        # Fast path for no special tokens
        print("Pretokenizing text (no special tokens)...")
        matches = list(get_pattern().finditer(text))
        
        result = Counter()
        for match in tqdm(matches, desc="Processing tokens", unit="tokens"):
            token_bytes = tuple(match.group(0).encode("utf-8"))
            result[token_bytes] += 1
        
        elapsed = time.time() - start_time
        print(f"Pretokenization completed in {elapsed:.2f}s")
        print(f"Found {len(result)} unique tokens from {sum(result.values())} total tokens")
        return result
    
    # Convert special tokens to set for faster lookup
    special_tokens_set = set(special_tokens)
    special_pattern = '|'.join(re.escape(token) for token in special_tokens)
    
    print("Splitting text on special tokens...")
    text_segments = re.split(f'({special_pattern})', text)
    
    result = Counter()
    print("Processing text segments...")
    for segment in tqdm(text_segments, desc="Processing segments", unit="segments"):
        if not segment:
            continue
        if segment in special_tokens_set:
            result[tuple(segment.encode("utf-8"))] += 1
        else:
            # Batch process segment tokens
            matches = list(get_pattern().finditer(segment))
            for match in matches:
                token_bytes = tuple(match.group(0).encode("utf-8"))
                result[token_bytes] += 1
    
    elapsed = time.time() - start_time
    print(f"Pretokenization completed in {elapsed:.2f}s")
    print(f"Found {len(result)} unique tokens from {sum(result.values())} total tokens")
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
    start_time = time.time()
    num_processes = mp.cpu_count()
    print(f"Starting parallel pretokenization with {num_processes} processes...")
    
    with open(filepath, "rb") as f:
        boundaries = find_chunk_boundaries(f, num_processes, "<|endoftext|>".encode("utf-8"))
    
    chunk_args = [(filepath, start, end, special_tokens) for start, end in zip(boundaries[:-1], boundaries[1:])]
    
    print(f"Processing {len(chunk_args)} chunks in parallel...")
    with mp.Pool(num_processes) as pool:
        chunk_results = []
        with tqdm(total=len(chunk_args), desc="Processing chunks", unit="chunks") as pbar:
            for result in pool.imap(process_chunk, chunk_args):
                chunk_results.append(result)
                pbar.update(1)
    
    print("Combining results from all chunks...")
    combined_result = Counter()
    for result in tqdm(chunk_results, desc="Combining chunks", unit="chunks"):
        combined_result.update(result)
    
    elapsed = time.time() - start_time
    print(f"Parallel pretokenization completed in {elapsed:.2f}s")
    print(f"Found {len(combined_result)} unique tokens from {sum(combined_result.values())} total tokens")
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
        start_time = time.time()
        print("Initializing pair counts from pretokens...")
        
        self.pair_counts.clear()
        self.pair_locations.clear()
        
        for pretoken, count in tqdm(self.pretoken_counts.items(), desc="Processing pretokens", unit="pretokens"):
            if pretoken in self.special_byte_tuples or len(pretoken) < 2:
                continue
            
            # Process all pairs in this pretoken at once
            pretoken_len = len(pretoken)
            for i in range(pretoken_len - 1):
                pair = (pretoken[i], pretoken[i + 1])
                self.pair_counts[pair] += count
                self.pair_locations[pair].add(pretoken)
        
        elapsed = time.time() - start_time
        print(f"Pair count initialization completed in {elapsed:.2f}s")
        print(f"Found {len(self.pair_counts)} unique pairs")

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
        start_time = time.time()
        print(f"Starting BPE training with {len(pretokens)} unique pretokens...")
        
        self.pretoken_counts = dict(pretokens)
        
        # Initialize pair counts once
        self._initialize_pair_counts()
        
        # Perform merges with incremental updates
        print(f"Performing up to {self.max_merges} merges...")
        merge_times = []
        
        with tqdm(total=self.max_merges, desc="BPE merges", unit="merges") as pbar:
            for step in range(self.max_merges):
                merge_start = time.time()
                
                best_pair = self._select_best_pair()
                if not best_pair:
                    print(f"No more pairs to merge at step {step}")
                    break
                
                # Record the merge
                self.vocab[self.next_token_id] = best_pair
                self.merges[best_pair] = self.next_token_id
                
                # Apply merge with incremental updates
                self._apply_merge_optimized(best_pair, self.next_token_id)
                self.next_token_id += 1
                
                merge_time = time.time() - merge_start
                merge_times.append(merge_time)
                
                # Update progress bar with current merge info
                pair_bytes = (self._get_token_bytes(best_pair[0]), self._get_token_bytes(best_pair[1]))
                pbar.set_postfix({
                    'merge_time': f'{merge_time:.3f}s',
                    'pairs_left': len(self.pair_counts),
                    'vocab_size': self.next_token_id
                })
                pbar.update(1)
        
        elapsed = time.time() - start_time
        avg_merge_time = sum(merge_times) / len(merge_times) if merge_times else 0
        
        print(f"BPE training completed in {elapsed:.2f}s")
        print(f"Completed {len(merge_times)} merges")
        print(f"Average time per merge: {avg_merge_time:.3f}s")
        print(f"Final vocabulary size: {self.next_token_id}")

    def get_vocab_and_merges(self):
        """Build and return final vocabulary and merges."""
        start_time = time.time()
        print("Building final vocabulary and merges...")
        
        output_vocab = {}
        for token_id in tqdm(self.vocab, desc="Building vocab", unit="tokens"):
            output_vocab[token_id] = self._get_token_bytes(token_id)
        
        output_merges = []
        for pair in tqdm(self.merges, desc="Building merges", unit="merges"):
            output_merges.append((output_vocab[pair[0]], output_vocab[pair[1]]))
        
        elapsed = time.time() - start_time
        print(f"Vocabulary and merges built in {elapsed:.2f}s")
        
        return output_vocab, output_merges

class TrainBPE:
    """Simplified BPE tokenizer interface with optimized training."""
    
    def __init__(self, input_path, vocab_size, special_tokens=None):
        assert vocab_size > 256, "Vocabulary size must be greater than 256"
        self.input_path = input_path
        self.vocab_size = vocab_size
        self.special_tokens = special_tokens or []
        self.trainer = OptimizedBPETrainer(vocab_size, special_tokens)

    def train(self):
        """Train the BPE tokenizer with optimized merging."""
        overall_start = time.time()
        print(f"Starting BPE tokenizer training for vocabulary size {self.vocab_size}")
        print(f"Input file: {self.input_path}")
        if self.special_tokens:
            print(f"Special tokens: {self.special_tokens}")
        
        # Skip multiprocessing for small files to avoid overhead
        file_size = os.path.getsize(self.input_path)
        print(f"File size: {file_size:,} bytes ({file_size / (1024*1024):.1f} MB)")
        
        if file_size > 50_000_000:  # 50MB threshold (increased from 10MB)
            print("Using parallel processing for large file...")
            pretokens = parallel_pretokenize(self.input_path, self.special_tokens)
        else:
            print("Using single-threaded processing for small file...")
            with open(self.input_path, "r", encoding="utf-8") as f:
                text = f.read()
            pretokens = pretokenize(text, self.special_tokens)
        
        # Train with optimized algorithm
        self.trainer.train(pretokens)
        
        # Get results
        self.output_vocab, self.output_merges = self.trainer.get_vocab_and_merges()
        
        overall_elapsed = time.time() - overall_start
        print(f"Complete BPE training finished in {overall_elapsed:.2f}s")
        print(f"Final vocabulary contains {len(self.output_vocab)} tokens")
        print(f"Generated {len(self.output_merges)} merge rules")

    def get_vocab_and_merges(self):
        """Return the trained vocabulary and merges."""
        return self.output_vocab, self.output_merges