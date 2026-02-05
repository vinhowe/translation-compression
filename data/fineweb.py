"""
FineWeb dataset (for srs pretraining)
https://huggingface.co/datasets/HuggingFaceFW/fineweb

example doc to highlight the structure of the dataset:
{
  "text": "Posted by mattsmith on 20th April 2012\nStraight from...",
  "id": "<urn:uuid:d853d453-196e-4488-a411-efc2b26c40d2>",
  "dump": "CC-MAIN-2013-20",
  "url": "http://nleastchatter.com/philliesphandom/tag/freddy-galvis/",
  "date": "2013-05-18T07:24:47Z",
  "file_path": "s3://commoncrawl/long.../path.../file.gz",
  "language": "en",
  "language_score": 0.9185474514961243,
  "token_count": 594
}

Example of downloading the 100B dataset of FineWeb, from root directory:
python dev/data/fineweb.py -v 100B
100B runs for small few hours, depending on your internet and computer.
"""

import os
import argparse
import multiprocessing as mp

import numpy as np
import tiktoken
from datasets import load_dataset
from tqdm import tqdm

from data_common import write_datafile

import xxhash
# ------------------------------------------

parser = argparse.ArgumentParser(description="FineWeb dataset preprocessing")
parser.add_argument(
    "-t", "--type", type=str, default="classic", help="Fineweb type, edu|classic"
)
parser.add_argument(
    "-v",
    "--version",
    type=str,
    default="10B",
    help="Fineweb data sample size, 10B|100B|350B",
)
parser.add_argument(
    "-s",
    "--shard_size",
    type=int,
    default=10**8,
    help="Size of each data shard in the output .bin files, in tokens",
)
parser.add_argument(
    "-o",
    "--output_dir",
    type=str,
    default=None,
    help="Optional output directory for the resulting .bin shards (overrides default cache dir)",
)
args = parser.parse_args()

# FineWeb has a few possible subsamples available
assert args.version in {"10B", "100B", "350B"}, (
    "version must be one of: 10B, 100B, 350B"
)
assert args.type in {"edu", "classic"}, "type must be one of: edu, classic"
directories = {
    ("classic", "10B"): ("fineweb10B", "sample-10BT"),
    ("classic", "100B"): ("fineweb100B", "sample-100BT"),
    ("classic", "350B"): ("fineweb350B", "sample-350BT"),
    ("edu", "10B"): ("edu_fineweb10B", "sample-10BT"),
    ("edu", "100B"): ("edu_fineweb100B", "sample-100BT"),
    ("edu", "350B"): ("edu_fineweb350B", "sample-350BT"),
}
local_dir, remote_name = directories[(args.type, args.version)]

# resolve output directory
default_cache_dir = os.path.join(os.path.dirname(__file__), local_dir)
DATA_CACHE_DIR = args.output_dir if args.output_dir is not None else default_cache_dir
os.makedirs(DATA_CACHE_DIR, exist_ok=True)

# download the dataset
fw = load_dataset(
    "HuggingFaceFW/fineweb", name=remote_name, split="train", streaming=True
)
name = "fineweb"


def tokenize_gpt2(doc):
    # tokenizes a single document and returns a numpy array of uint16 tokens
    enc = tiktoken.get_encoding("gpt2")

    def encode(s):
        return enc.encode_ordinary(s)

    eot = enc._special_tokens["<|endoftext|>"]  # end of text token
    tokens = [eot]  # the special <|endoftext|> token delimits all documents
    tokens.extend(encode(doc["text"]))
    tokens_np = np.array(tokens)
    assert (0 <= tokens_np).all() and (tokens_np < 2**16).all(), (
        "token dictionary too large for uint16"
    )
    tokens_np_uint = tokens_np.astype(np.uint16)
    return tokens_np_uint


CHUNK_SIZE = 32  # tokens per chunk for hashing/dedup and writing

# tokenize all documents and write output shards composed of deduplicated 32-token chunks
assert args.shard_size % CHUNK_SIZE == 0, (
    f"shard_size must be a multiple of {CHUNK_SIZE} to align with chunk boundaries"
)

cpu_total = os.cpu_count() or 1
nprocs = max(1, cpu_total - 2)  # don't hog the entire system
with mp.Pool(nprocs) as pool:
    shard_index = 0
    # preallocate buffer to hold current shard (always filled in multiples of CHUNK_SIZE)
    all_tokens_np = np.empty((args.shard_size,), dtype=np.uint16)
    token_count = 0
    progress_bar = None

    # buffer for tokens that don't yet make up a full 32-token chunk across document boundaries
    pending_np = np.empty((0,), dtype=np.uint16)
    # store hashes as 16-byte digests (bytes) for memory efficiency
    seen_hashes = set()

    # main processing loop: tokenize docs in parallel, then chunk/dedupe serially
    tokenize = tokenize_gpt2
    for tokens in pool.imap(tokenize, fw, chunksize=16):
        # combine with pending tokens from previous step to form a continuous stream
        if pending_np.size:
            combined = np.concatenate([pending_np, tokens])
        else:
            combined = tokens
        # compute how many full 32-token chunks we can form
        num_full_chunks = combined.size // CHUNK_SIZE
        if num_full_chunks:
            chunkable = combined[: num_full_chunks * CHUNK_SIZE]
            chunks_matrix = chunkable.reshape(-1, CHUNK_SIZE)
            # iterate over chunks, hash, dedupe, and append only unseen chunks
            for i in range(chunks_matrix.shape[0]):
                chunk_row = chunks_matrix[i]
                # ensure deterministic bytes for hashing: little-endian 16-bit
                chunk_bytes = chunk_row.astype(np.dtype("<u2"), copy=False).tobytes()
                digest = xxhash.xxh3_128_digest(chunk_bytes)
                if digest in seen_hashes:
                    continue
                seen_hashes.add(digest)

                # if current shard is full, write it and start a new shard
                if token_count + CHUNK_SIZE > args.shard_size:
                    split = "val" if shard_index == 0 else "train"
                    filename = os.path.join(
                        DATA_CACHE_DIR, f"{name}_{split}_{shard_index:06d}.bin"
                    )
                    if progress_bar is not None:
                        # fill remaining tokens in bar to complete the shard
                        progress_bar.close()
                        progress_bar = None
                    write_datafile(
                        filename, (all_tokens_np[:token_count]).tolist(), "gpt-2"
                    )
                    shard_index += 1
                    token_count = 0

                # append chunk to current shard
                if progress_bar is None:
                    progress_bar = tqdm(
                        total=args.shard_size,
                        unit="tokens",
                        desc=f"Shard {shard_index}",
                    )
                all_tokens_np[token_count : token_count + CHUNK_SIZE] = chunk_row
                token_count += CHUNK_SIZE
                progress_bar.update(CHUNK_SIZE)
        # carry over remaining tokens (< CHUNK_SIZE) to next iteration
        pending_np = combined[num_full_chunks * CHUNK_SIZE :]

    # after all docs are processed, write any remaining shard content (must be multiple of CHUNK_SIZE)
    if token_count != 0:
        split = "val" if shard_index == 0 else "train"
        filename = os.path.join(DATA_CACHE_DIR, f"{name}_{split}_{shard_index:06d}.bin")
        write_datafile(filename, (all_tokens_np[:token_count]).tolist(), "gpt-2")
