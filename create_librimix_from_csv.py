#!/usr/bin/env python3
# scripts/create_librimix_from_csv.py
# Build Libri(2|3)Mix from user-provided CSVs (train-clean-100, dev-clean, test-clean)
# No WHAM/noise; CSV is the source of truth for utterance pairing & gains.
#
# Based on the original repo structure & functions but with WHAM removed
# and CSV-driven subset control. See originals for reference:
# - scripts/create_librimix_from_metadata.py
# - scripts/create_librimix_metadata.py

import os
import argparse
import pandas as pd
import numpy as np
import soundfile as sf
from scipy.signal import resample_poly
from functools import partial
import tqdm.contrib.concurrent

EPS = 1e-10
RATE = 16000  # LibriSpeech native rate

# ---------------------------
# Args
# ---------------------------
def build_parser():
    p = argparse.ArgumentParser(description="Create Libri(2|3)Mix from CSVs only (no WHAM).")
    p.add_argument('--librispeech_dir', type=str, required=True,
                   help='Path to LibriSpeech root (contains dev-clean, test-clean, train-clean-100)')
    p.add_argument('--csv_dir', type=str, required=True,
                   help='Directory containing the three CSVs: '
                        'libri2mix_train-clean-100.csv, libri2mix_dev-clean.csv, libri2mix_test-clean.csv')
    p.add_argument('--librimix_outdir', type=str, default=None,
                   help='Dataset root directory to write Libri{n}Mix (default: parent of csv_dir)')
    p.add_argument('--n_src', type=int, required=True, choices=[2, 3],
                   help='Number of sources in mixtures (2 or 3)')
    p.add_argument('--freqs', nargs='+', default=['8k', '16k'],
                   help='Frequencies to generate, e.g., --freqs 8k 16k (case-insensitive)')
    p.add_argument('--modes', nargs='+', default=['min', 'max'],
                   help='Crop/pad modes to generate, e.g., --modes min max')
    p.add_argument('--types', nargs='+', default=['mix_clean'],
                   help='Kept for compatibility; only mix_clean is supported (no noise).')
    p.add_argument('--subset_patterns', nargs='+',
                   default=['train-clean-100', 'dev-clean', 'test-clean'],
                   help='Which CSVs to process by filename pattern. Defaults to exactly the three required subsets.')
    return p

# ---------------------------
# Core build
# ---------------------------
def main(args):
    librispeech_dir = args.librispeech_dir
    csv_dir = args.csv_dir
    n_src = args.n_src

    freqs = [f.lower() for f in args.freqs]
    modes = [m.lower() for m in args.modes]
    types = [t.lower() for t in args.types]
    if types != ['mix_clean']:
        print("[warn] Only 'mix_clean' is supported in this minimal builder. Forcing to ['mix_clean'].")
        types = ['mix_clean']

    # Decide output root
    out_root = args.librimix_outdir or os.path.dirname(os.path.abspath(csv_dir))
    out_root = os.path.join(out_root, f'Libri{n_src}Mix')

    # Find subset CSVs
    all_csvs = [f for f in os.listdir(csv_dir) if f.endswith('.csv') and '_info' not in f]
    target_csvs = [f for f in all_csvs if any(pat in f for pat in args.subset_patterns)]
    if not target_csvs:
        raise FileNotFoundError(f"No target CSVs found under {csv_dir} for patterns {args.subset_patterns}")

    # Process each subset CSV
    for md_filename in sorted(target_csvs):
        csv_path = os.path.join(csv_dir, md_filename)
        process_subset(csv_path, freqs, modes, types, n_src, librispeech_dir, out_root)

def process_subset(csv_path, freqs, modes, types, n_src, librispeech_dir, out_root):
    md = pd.read_csv(csv_path, engine='python')
    # Normalize and validate required columns
    needed_paths = [f"source_{i+1}_path" for i in range(n_src)]
    for col in needed_paths:
        if col not in md.columns:
            raise ValueError(f"CSV {os.path.basename(csv_path)} missing required column: {col}")

    # Gains are optional; if missing use 1.0
    gains_cols = []
    for i in range(n_src):
        gname = f"source_{i+1}_gain"
        if gname in md.columns:
            gains_cols.append(gname)
        else:
            md[gname] = 1.0
            gains_cols.append(gname)

    # Prepare directory names like the original code
    # e.g., from 'libri2mix_dev-clean.csv' -> subset dir 'dev'
    subset_dir = os.path.basename(csv_path).replace('.csv', '')
    for token in ['libri2mix_', 'libri3mix_', 'libri2mix', 'libri3mix', '-clean']:
        subset_dir = subset_dir.replace(token, '')
    # Keep only known names
    if 'train-clean-100' in os.path.basename(csv_path):
        subset_dir = 'train-100'
    elif 'dev-clean' in os.path.basename(csv_path):
        subset_dir = 'dev'
    elif 'test-clean' in os.path.basename(csv_path):
        subset_dir = 'test'

    for freq_str in freqs:
        freq_dir = os.path.join(out_root, 'wav' + freq_str)
        freq = int(freq_str.strip('k')) * 1000

        for mode in modes:
            mode_dir = os.path.join(freq_dir, mode)
            subset_out = os.path.join(mode_dir, subset_dir)

            # Make subdirs: s1..sN and mix_clean (no noise)
            subdirs = [f"s{i+1}" for i in range(n_src)] + ['mix_clean']
            for sd in subdirs:
                os.makedirs(os.path.join(subset_out, sd), exist_ok=True)

            # Metadata dir (matches original layout)
            meta_dir = os.path.join(mode_dir, 'metadata')
            os.makedirs(meta_dir, exist_ok=True)

            # Dataframes to mirror original outputs
            metrics_df = create_empty_metrics_md(n_src, 'mix_clean')  # only mix_clean
            mixture_df = create_empty_mixture_md(n_src, 'mix_clean')

            # Row processing in parallel
            process_fn = partial(
                process_row_build,
                librispeech_dir=librispeech_dir,
                n_src=n_src,
                freq=freq,
                mode=mode,
                subset_out=subset_out
            )
            for result in tqdm.contrib.concurrent.process_map(process_fn, [r for _, r in md.iterrows()], chunksize=10):
                mix_id, snr_list, abs_mix_path, abs_source_paths, sig_len = result
                add_to_metrics_metadata(metrics_df, mix_id, snr_list)
                add_to_mixture_metadata(mixture_df, mix_id, abs_mix_path, abs_source_paths, [], sig_len, 'mix_clean')

            # Save metadata
            base = f"{os.path.basename(subset_out)}_mix_clean"
            mixture_df.to_csv(os.path.join(meta_dir, f"mixture_{base}.csv"), index=False)
            metrics_df.to_csv(os.path.join(meta_dir, f"metrics_{base}.csv"), index=False)

def process_row_build(row, librispeech_dir, n_src, freq, mode, subset_out):
    mix_id = str(row.get('mixture_ID', build_mix_id_from_paths(row, n_src)))

    # Read sources, apply gains
    sources = []
    for i in range(n_src):
        rel = row[f"source_{i+1}_path"]
        g = float(row.get(f"source_{i+1}_gain", 1.0))
        wav, _ = sf.read(os.path.join(librispeech_dir, rel), dtype='float32')
        wav = wav * g
        sources.append(wav)

    # Resample
    sources = [resample_poly(s, freq, RATE) for s in sources]

    # Fit lengths
    if mode == 'min':
        target = min(len(s) for s in sources)
        sources = [s[:target] for s in sources]
    else:
        target = max(len(s) for s in sources)
        sources = [np.pad(s, (0, target - len(s)), mode='constant') for s in sources]

    # Write sources
    src_paths = []
    ex = mix_id + '.wav'
    for i, sig in enumerate(sources):
        p = os.path.join(subset_out, f"s{i+1}", ex)
        ap = os.path.abspath(p)
        sf.write(ap, sig, freq)
        src_paths.append(ap)

    # Mix (clean only)
    mix = np.zeros_like(sources[0])
    for s in sources:
        mix += s
    mix_path = os.path.abspath(os.path.join(subset_out, 'mix_clean', ex))
    sf.write(mix_path, mix, freq)

    # Compute SNRs of each source vs (mix - source)
    snrs = []
    for i in range(n_src):
        resid = mix - sources[i]
        snr = 10 * np.log10((np.mean(sources[i] ** 2) + EPS) / (np.mean(resid ** 2) + EPS) + EPS)
        snrs.append(snr)

    return mix_id, snrs, mix_path, src_paths, len(mix)

def build_mix_id_from_paths(row, n_src):
    # Fallback if mixture_ID isn't in CSV: derive from filenames (stable & readable)
    ids = []
    for i in range(n_src):
        stem = os.path.basename(str(row[f"source_{i+1}_path"])).replace('.flac', '').replace('.wav', '')
        ids.append(stem)
    return "_".join(ids)

# ---------------------------
# Metadata helpers (mirrors original)
# ---------------------------
def create_empty_metrics_md(n_src, subdir):
    df = pd.DataFrame()
    df['mixture_ID'] = {}
    for i in range(n_src):
        df[f"source_{i + 1}_SNR"] = {}
    return df

def create_empty_mixture_md(n_src, subdir):
    df = pd.DataFrame()
    df['mixture_ID'] = {}
    df['mixture_path'] = {}
    for i in range(n_src):
        df[f"source_{i + 1}_path"] = {}
    df['length'] = {}
    return df

def add_to_metrics_metadata(df, mixture_id, snr_list):
    df.loc[len(df)] = [mixture_id] + snr_list

def add_to_mixture_metadata(df, mix_id, abs_mix_path, abs_sources_path,
                            _abs_noise_path_unused, length, _subdir_unused):
    row = [mix_id, abs_mix_path] + abs_sources_path + [length]
    df.loc[len(df)] = row

if __name__ == "__main__":
    args = build_parser().parse_args()
    main(args)
