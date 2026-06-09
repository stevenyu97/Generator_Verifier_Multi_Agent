#!/usr/bin/env python3
"""Download raw FinQA and ConvFinQA JSON files into llm/convfinqa/raw/."""
from __future__ import annotations

import argparse
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent
RAW = ROOT / "raw"

FINQA_URLS = {
    "train.json": "https://raw.githubusercontent.com/czyssrs/FinQA/master/dataset/train.json",
    "dev.json": "https://raw.githubusercontent.com/czyssrs/FinQA/master/dataset/dev.json",
    "test.json": "https://raw.githubusercontent.com/czyssrs/FinQA/master/dataset/test.json",
}

CONVFINQA_ZIP_URL = (
    "https://raw.githubusercontent.com/czyssrs/ConvFinQA/main/data.zip"
)


def _download(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 0:
        print(f"[skip] {dest} already exists")
        return
    print(f"[download] {url} -> {dest}")
    urllib.request.urlretrieve(url, dest)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--datasets",
        nargs="+",
        choices=["finqa", "convfinqa", "all"],
        default=["all"],
        help="Which raw datasets to fetch",
    )
    cli = p.parse_args()
    want = set(cli.datasets)
    if "all" in want:
        want = {"finqa", "convfinqa"}

    if "finqa" in want:
        finqa_dir = RAW / "finqa"
        for name, url in FINQA_URLS.items():
            _download(url, finqa_dir / name)

    if "convfinqa" in want:
        cfq_dir = RAW / "convfinqa"
        zip_path = cfq_dir / "data.zip"
        _download(CONVFINQA_ZIP_URL, zip_path)
        marker = cfq_dir / ".extracted"
        if not marker.exists():
            print(f"[extract] {zip_path} -> {cfq_dir}")
            with zipfile.ZipFile(zip_path, "r") as zf:
                zf.extractall(cfq_dir)
            marker.write_text("ok", encoding="utf-8")
        else:
            print(f"[skip] ConvFinQA zip already extracted")

    print(f"[done] raw files under {RAW}")


if __name__ == "__main__":
    main()
