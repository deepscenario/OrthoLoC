#!/bin/bash
# Download the full OrthoLoC dataset from TUM servers
# Usage: ./download_dataset.sh [output_dir] [num_parallel]
BASE_URL="https://cvg.cit.tum.de/webshare/g/papers/Dhaouadi/OrthoLoC/full"
OUT_DIR="${1:-./OrthoLoC_dataset}"
JOBS="${2:-4}"  # number of parallel downloads (default: 4)

for split in train val test_inPlace test_outPlace; do
    mkdir -p "$OUT_DIR/$split"
    echo "Downloading $split ($JOBS parallel) ..."
    # Fetch file listing, extract .npz links, download in parallel
    wget -q -O - "$BASE_URL/$split/" \
        | grep -oP 'href="\K[^"]*\.npz' \
        | xargs -P "$JOBS" -I {} wget -q --show-progress -nc -P "$OUT_DIR/$split" "$BASE_URL/$split/{}"
done

echo "Done. Dataset saved to $OUT_DIR"