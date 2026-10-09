"""Index the raw dataset, carve out a patient-level validation split and cache resized images.

    python scripts/prepare_data.py --raw . --out data

Outputs:
    data/manifest.csv      one row per image (path, split, label, subtype, patient_id); row i <-> images[i]
                           (the stored val split uses seed 42; training re-draws it from --split-seed)
    data/images.npy        (N, 256, 256) uint8 grayscale cache
    data/data_report.json  split sizes, patient counts and leakage checks
"""

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path

from pneumonia.data import assign_val_split, build_manifest, cache_images


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--raw", default=".", help="folder containing train/ and test/")
    p.add_argument("--out", default="data")
    p.add_argument("--size", type=int, default=256)
    p.add_argument("--workers", type=int, default=8)
    args = p.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    df = assign_val_split(build_manifest(args.raw))

    patients = {s: set(df.loc[df["split"] == s, "patient_id"]) for s in ("train", "val", "test")}
    assert not patients["train"] & patients["val"], "patient leakage between train and val"

    hashes = defaultdict(set)
    for path, split in zip(df["path"], df["split"], strict=True):
        hashes[hashlib.md5((Path(args.raw) / path).read_bytes()).hexdigest()].add(split)
    report = {
        "images": df.groupby(["split", "label"]).size().unstack().rename(columns={0: "NORMAL", 1: "PNEUMONIA"}).to_dict("index"),
        "patients": {s: len(v) for s, v in patients.items()},
        "train_val_patient_overlap": 0,
        "train_test_patient_id_overlap": len((patients["train"] | patients["val"]) & patients["test"]),
        "exact_duplicate_images": len(df) - len(hashes),
        "exact_duplicates_across_splits": sum(1 for s in hashes.values() if len(s) > 1),
    }
    (out / "data_report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))

    df.to_csv(out / "manifest.csv", index=False)
    print(f"caching {len(df)} images at {args.size}x{args.size} ...")
    cache_images(df, args.raw, out / "images.npy", args.size, args.workers)
    print(f"done -> {out}/")


if __name__ == "__main__":
    main()
