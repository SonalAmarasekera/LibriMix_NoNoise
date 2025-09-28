#!/usr/bin/env bash
# scripts/generate_librimix_minimal.sh
set -eu

storage_dir="$1"          # e.g., /data
csv_dir="$2"              # e.g., /data/LibriMix/metadata/csv
#n_src="${3:-2}"
n_src=2# 2 or 3
python_path="${PYTHON:-python}"

librispeech_dir="$storage_dir/LibriSpeech"
mkdir -p "$storage_dir"

download_split () {
  local split="$1"   # dev-clean | test-clean | train-clean-100
  if [ ! -d "$librispeech_dir/$split" ]; then
    echo "Downloading LibriSpeech/$split into $storage_dir"
    url="http://www.openslr.org/resources/12/${split}.tar.gz"
    wget -c --tries=0 --read-timeout=20 "$url" -P "$storage_dir"
    tar -xzf "$storage_dir/${split}.tar.gz" -C "$storage_dir"
    rm -f "$storage_dir/${split}.tar.gz"
  fi
}

download_split dev-clean
download_split test-clean
download_split train-clean-100

# Build only train-100, dev, test from provided CSVs (no WHAM)
$python_path scripts/create_librimix_from_csv.py \
  --librispeech_dir "$librispeech_dir" \
  --csv_dir "$csv_dir" \
  --librimix_outdir "$storage_dir" \
  --n_src "$n_src" \
  --freqs 16k \
  --modes min \
  --types mix_clean

