#!/usr/bin/env python3
"""
Train BPE tokenizer on TinyStories dataset with profiling.
"""

import time
import cProfile
import pstats
import psutil
import os
import json
import base64
import sys
from pathlib import Path
from cs336_basics.train_bpe import TrainBPE

def train_and_profile(data_path, vocab_size=10000):
    """Train BPE tokenizer with profiling and analysis."""
    
    if not Path(data_path).exists():
        print(f"Error: File not found at {data_path}")
        return
    
    # Memory and time tracking
    process = psutil.Process(os.getpid())
    
    # Initialize tokenizer
    tokenizer = TrainBPE(data_path, vocab_size=vocab_size, special_tokens=["<|endoftext|>"])
    
    # Profile training
    profiler = cProfile.Profile()
    start_time = time.time()
    profiler.enable()
    tokenizer.train()
    profiler.disable()
    end_time = time.time()
    
    # Get results
    vocab, merges = tokenizer.get_vocab_and_merges()
    peak_memory = process.memory_info().rss / (1024**3)
    training_time = end_time - start_time
    
    # Save results
    os.makedirs("output", exist_ok=True)
    with open("output/vocab.json", 'w') as f:
        vocab_data = {str(k): base64.b64encode(v).decode('utf-8') for k, v in vocab.items()}
        json.dump(vocab_data, f)
    
    with open("output/merges.txt", 'w') as f:
        for left, right in merges:
            f.write(f"{base64.b64encode(left).decode('utf-8')} {base64.b64encode(right).decode('utf-8')}\n")
    
    # Find longest token
    longest_token = max(vocab.values(), key=len)
    decoded = longest_token.decode('utf-8', errors='replace')
    
    # Profiling analysis
    stats = pstats.Stats(profiler)
    top_func = max(stats.get_stats_profile().func_profiles.items(), 
                   key=lambda x: x[1].cumtime)
    func_name = top_func[0][2]  # function name
    func_time_pct = (top_func[1].cumtime / training_time) * 100
    
    # Deliverable answers
    print(f"(a) Training took {training_time/60:.2f} minutes and used {peak_memory:.1f}GB peak memory. "
          f"The longest token is {len(longest_token)} bytes ('{decoded[:30]}...') which "
          f"{'makes sense' if decoded.isprintable() and decoded.strip() else 'does not make sense'} "
          f"as a common text sequence.")
    
    print(f"(b) Profiling shows that the most time-consuming part is {func_name} "
          f"which took {func_time_pct:.1f}% of the total training time.")

if __name__ == "__main__":
    if len(sys.argv) < 2 or len(sys.argv) > 3:
        print("Usage: python train_tinystories.py <data_path> [vocab_size]")
        print("Examples:")
        print("  python train_bpe_file.py /Users/kennethlee/Documents/assignment1-basics/data/TinyStoriesV2-GPT4-train.txt")
        print("  python train_bpe_file.py /Users/kennethlee/Documents/assignment1-basics/data/TinyStoriesV2-GPT4-train.txt 5000")
        print("  python train_bpe_file.py /Users/kennethlee/Documents/assignment1-basics/data/TinyStoriesV2-GPT4-train.txt 20000")
        sys.exit(1)
    
    data_path = sys.argv[1]
    vocab_size = int(sys.argv[2]) if len(sys.argv) == 3 else 10000
    
    print(f"Training BPE tokenizer with vocab_size={vocab_size}")
    train_and_profile(data_path, vocab_size)