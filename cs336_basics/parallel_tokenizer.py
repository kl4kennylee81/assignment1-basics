import os
from typing import BinaryIO
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np
from tqdm import tqdm

from cs336_basics.tokenizer import Tokenizer

def find_chunk_boundaries(
    file: BinaryIO, 
    desired_num_chunks: int, 
    split_special_token: bytes
) -> list[int]:
    """
    Chunk the file into parts that can be counted independently.
    May return fewer chunks if the boundaries end up overlapping.
    """
    assert isinstance(split_special_token, bytes), (
        "Must represent special token as a bytestring"
    )

    # Get total file size in bytes
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)

    chunk_size = file_size // desired_num_chunks

    # Initial guesses for chunk boundary locations, uniformly spaced
    # Chunks start on previous index, don't include last index
    chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
    chunk_boundaries[-1] = file_size

    mini_chunk_size = 4096  # Read ahead by 4k bytes at a time

    for bi in range(1, len(chunk_boundaries) - 1):
        initial_position = chunk_boundaries[bi]
        file.seek(initial_position)  # Start at boundary guess
        while True:
            mini_chunk = file.read(mini_chunk_size)  # Read a mini chunk

            # If EOF, this boundary should be at the end of the file
            if mini_chunk == b"":
                chunk_boundaries[bi] = file_size
                break

            # Find the special token in the mini chunk
            found_at = mini_chunk.find(split_special_token)
            if found_at != -1:
                chunk_boundaries[bi] = initial_position + found_at
                break
            initial_position += mini_chunk_size

    # Make sure all boundaries are unique, but might be fewer than desired_num_chunks
    return sorted(set(chunk_boundaries))

def encode_chunk_worker(args):
    """
    Worker function to encode a single chunk.
    Returns (chunk_index, encoded_tokens)
    """
    chunk_index, file_path, start_pos, end_pos, vocab_path, merges_path, special_tokens = args
    
    # Load tokenizer
    tokenizer = Tokenizer.from_files(vocab_path, merges_path, special_tokens)
    
    # Read the chunk
    with open(file_path, 'rb') as f:
        f.seek(start_pos)
        chunk_bytes = f.read(end_pos - start_pos)
        chunk_text = chunk_bytes.decode('utf-8', errors='ignore')
    
    # Use the faster direct encode method instead of encode_iterable
    encoded = tokenizer.encode(chunk_text)
    
    return chunk_index, encoded

def parallel_encode_with_boundaries(
    file_path: str,
    split_special_token: str = "<|endoftext|>",
    vocab_path: str = "tinystories_vocab.json",
    merges_path: str = "tinystories_merges.txt", 
    special_tokens: list = None,
    num_workers: int = None,
    desired_num_chunks: int = None,
    checkpoint_dir: str = "output/chunks",
    checkpoint_interval: int = 10
) -> list[int]:
    """
    Encode a large text file in parallel using special token boundaries.
    Saves progress checkpoints every N completed chunks.
    
    Args:
        file_path: Path to text file to encode
        split_special_token: Special token to use as chunk boundary  
        vocab_path: Path to vocabulary file
        merges_path: Path to merges file
        special_tokens: List of special tokens for tokenizer
        num_workers: Number of worker processes (default: CPU count)
        desired_num_chunks: Target number of chunks (default: 4x workers)
        checkpoint_dir: Directory to save progress checkpoints
        checkpoint_interval: Save checkpoint every N completed chunks
    
    Returns:
        List of encoded token IDs
    """
    if special_tokens is None:
        special_tokens = ["<|endoftext|>"]
    
    if num_workers is None:
        num_workers = min(mp.cpu_count(), 8)
    
    if desired_num_chunks is None:
        desired_num_chunks = num_workers * 4
    
    # Create checkpoint directory
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    print(f"Finding chunk boundaries using '{split_special_token}'...")
    
    # Find chunk boundaries
    with open(file_path, 'rb') as f:
        boundaries = find_chunk_boundaries(
            f, desired_num_chunks, split_special_token.encode('utf-8')
        )
    
    num_actual_chunks = len(boundaries) - 1
    print(f"Found {num_actual_chunks} chunks (requested {desired_num_chunks})")
    
    # Check for existing progress checkpoint
    progress_file = os.path.join(checkpoint_dir, "progress.npy")
    completed_chunks = {}  # chunk_index -> encoded_tokens
    start_chunk = 0
    
    if os.path.exists(progress_file):
        print(f"Loading existing progress from {progress_file}...")
        progress_data = np.load(progress_file, allow_pickle=True).item()
        completed_chunks = progress_data.get('completed_chunks', {})
        start_chunk = len(completed_chunks)
        print(f"Resuming from chunk {start_chunk} ({len(completed_chunks)} chunks already completed)")
    
    # Prepare worker arguments for remaining chunks
    worker_args = []
    for i in range(start_chunk, num_actual_chunks):
        start_pos = boundaries[i]
        end_pos = boundaries[i + 1]
        chunk_size_mb = (end_pos - start_pos) / (1024 * 1024)
        
        args = (i, file_path, start_pos, end_pos, vocab_path, merges_path, special_tokens)
        worker_args.append(args)
        
        if i < start_chunk + 10:  # Show first 10 remaining chunks
            print(f"Chunk {i}: {start_pos:,} - {end_pos:,} ({chunk_size_mb:.1f}MB)")
    
    chunks_to_process = len(worker_args)
    if chunks_to_process > 0:
        print(f"\nProcessing {chunks_to_process} remaining chunks with {num_workers} workers...")
        print(f"Will save progress every {checkpoint_interval} completed chunks")
        
        chunks_completed_since_last_save = 0
        
        # Process remaining chunks in parallel
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            # Submit remaining chunks
            future_to_chunk = {
                executor.submit(encode_chunk_worker, args): args[0]  # args[0] is chunk_index
                for args in worker_args
            }
            
            # Collect results with progress tracking
            with tqdm(total=chunks_to_process, desc="Encoding chunks", unit="chunk") as pbar:
                for future in as_completed(future_to_chunk):
                    chunk_idx = future_to_chunk[future]
                    try:
                        chunk_index, encoded_tokens = future.result()
                        
                        # Store completed chunk
                        completed_chunks[chunk_index] = encoded_tokens
                        chunks_completed_since_last_save += 1
                        
                        pbar.update(1)
                        pbar.set_postfix({
                            'chunk': chunk_index, 
                            'tokens': f"{len(encoded_tokens):,}",
                            'completed': len(completed_chunks),
                            'next_save': checkpoint_interval - chunks_completed_since_last_save
                        })
                        
                        # Save progress checkpoint every N chunks
                        if chunks_completed_since_last_save >= checkpoint_interval:
                            progress_data = {
                                'completed_chunks': completed_chunks,
                                'total_chunks': num_actual_chunks,
                                'file_path': file_path,
                                'boundaries': boundaries
                            }
                            np.save(progress_file, progress_data)
                            chunks_completed_since_last_save = 0
                            print(f"\n✓ Progress saved: {len(completed_chunks)}/{num_actual_chunks} chunks completed")
                        
                    except Exception as exc:
                        print(f'Chunk {chunk_idx} generated exception: {exc}')
                        raise
        
        # Save final progress if any chunks completed since last save
        if chunks_completed_since_last_save > 0:
            progress_data = {
                'completed_chunks': completed_chunks,
                'total_chunks': num_actual_chunks,
                'file_path': file_path,
                'boundaries': boundaries
            }
            np.save(progress_file, progress_data)
            print(f"✓ Final progress saved: {len(completed_chunks)}/{num_actual_chunks} chunks completed")
    
    else:
        print("All chunks already completed!")
    
    # Merge all chunks in order
    print("\nMerging all completed chunks...")
    all_tokens = []
    total_tokens = 0
    
    for i in range(num_actual_chunks):
        if i in completed_chunks:
            chunk_tokens = completed_chunks[i]
            all_tokens.extend(chunk_tokens)
            total_tokens += len(chunk_tokens)
        else:
            print(f"Warning: Missing chunk {i}")
    
    print(f"Merged total: {total_tokens:,} tokens from {len(completed_chunks)} chunks")
    
    # Optionally clean up progress file
    cleanup = input("Delete progress checkpoint file? (y/N): ").lower().strip()
    if cleanup == 'y':
        os.remove(progress_file)
        print(f"Deleted {progress_file}")
        try:
            os.rmdir(checkpoint_dir)
            print(f"Removed empty directory {checkpoint_dir}")
        except OSError:
            pass  # Directory not empty
    
    return all_tokens

def encode_file_parallel(file_path: str, output_path: str = None):
    """
    Complete function to encode a file and save results.
    """
    if output_path is None:
        output_path = file_path.replace('.txt', '_encoded_parallel.npy')
    
    # Encode in parallel
    encoded_tokens = parallel_encode_with_boundaries(
        file_path=file_path,
        split_special_token="<|endoftext|>",  # TinyStories uses this
        num_workers=4,  # Adjust based on your CPU
        desired_num_chunks=100  # More chunks = better load balancing
    )
    
    # Save to numpy array
    print(f"Saving {len(encoded_tokens):,} tokens to {output_path}...")
    encoded_array = np.array(encoded_tokens, dtype=np.uint16)
    np.save(output_path, encoded_array)
    
    # File size info
    file_size = os.path.getsize(output_path)
    print(f"Saved to {output_path} ({file_size:,} bytes, {file_size/(1024*1024):.1f}MB)")
    
    return encoded_tokens