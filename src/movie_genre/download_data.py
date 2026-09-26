"""Download or copy the movie genre classification data files."""

from __future__ import annotations

import argparse
import shutil
import urllib.request
from pathlib import Path

from movie_genre.config import DATA_DIR, DATA_FILES


def copy_from_source_dir(source_dir: Path, destination_dir: Path) -> None:
    for name in DATA_FILES:
        source_file = source_dir / name
        destination_file = destination_dir / name
        if not source_file.exists():
            raise FileNotFoundError(f"Missing source file: {source_file}")
        shutil.copy2(source_file, destination_file)
        print(f"Copied {name} -> {destination_file}")


def download_from_base_url(base_url: str, destination_dir: Path) -> None:
    for name in DATA_FILES:
        url = f"{base_url.rstrip('/')}/{name}"
        destination_file = destination_dir / name
        urllib.request.urlretrieve(url, destination_file)
        print(f"Downloaded {name} -> {destination_file}")


def main() -> int:
    parser = argparse.ArgumentParser(prog="movie-genre-download", description="Copy or download the movie genre data files.")
    parser.add_argument("--source-dir", type=Path, help="Copy the data files from a local directory.")
    parser.add_argument("--base-url", help="Download the data files from a remote base URL.")
    parser.add_argument("--output-dir", type=Path, default=DATA_DIR, help="Where to place the files (default: data/raw).")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    if args.source_dir:
        copy_from_source_dir(args.source_dir, args.output_dir)
        return 0

    if args.base_url:
        download_from_base_url(args.base_url, args.output_dir)
        return 0

    print("Provide either --source-dir or --base-url.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
