"""Download or copy the movie genre classification data files and verify their checksums."""

from __future__ import annotations

import argparse
import hashlib
import shutil
import urllib.request
from pathlib import Path

from movie_genre.config import DATA_DIR, DATA_FILES, DATA_SHA256


def file_sha256(path: Path) -> str:
    """SHA-256 of the file with CRLF line endings normalized to LF."""
    return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def verify_file(path: Path, expected: str) -> None:
    """Raise ValueError if the file does not match the expected checksum."""
    actual = file_sha256(path)
    if actual != expected:
        raise ValueError(f"Checksum mismatch for {path}: expected {expected}, got {actual}")


def copy_from_source_dir(source_dir: Path, destination_dir: Path, verify: bool = True) -> None:
    for name in DATA_FILES:
        source_file = source_dir / name
        destination_file = destination_dir / name
        if not source_file.exists():
            raise FileNotFoundError(f"Missing source file: {source_file}")
        if verify:
            verify_file(source_file, DATA_SHA256[name])
        shutil.copy2(source_file, destination_file)
        print(f"Copied {name} -> {destination_file}")


def download_from_base_url(base_url: str, destination_dir: Path, verify: bool = True) -> None:
    for name in DATA_FILES:
        url = f"{base_url.rstrip('/')}/{name}"
        destination_file = destination_dir / name
        # Download next to the target and only replace it once the checksum matches.
        partial_file = destination_file.with_name(name + ".part")
        urllib.request.urlretrieve(url, partial_file)
        if verify:
            try:
                verify_file(partial_file, DATA_SHA256[name])
            except ValueError:
                partial_file.unlink()
                raise
        partial_file.replace(destination_file)
        print(f"Downloaded {name} -> {destination_file}")


def verify_dir(data_dir: Path) -> None:
    for name in DATA_FILES:
        verify_file(data_dir / name, DATA_SHA256[name])
        print(f"OK {name}")


def main() -> int:
    parser = argparse.ArgumentParser(prog="movie-genre-download", description="Copy or download the movie genre data files.")
    parser.add_argument("--source-dir", type=Path, help="Copy the data files from a local directory.")
    parser.add_argument("--base-url", help="Download the data files from a remote base URL.")
    parser.add_argument("--output-dir", type=Path, default=DATA_DIR, help="Where to place the files (default: data/raw).")
    parser.add_argument("--verify-only", action="store_true", help="Only check the SHA-256 of the files in --output-dir.")
    parser.add_argument("--skip-verify", action="store_true", help="Do not check SHA-256 checksums (for a different dataset).")
    args = parser.parse_args()

    if args.verify_only:
        verify_dir(args.output_dir)
        return 0

    args.output_dir.mkdir(parents=True, exist_ok=True)
    verify = not args.skip_verify

    if args.source_dir:
        copy_from_source_dir(args.source_dir, args.output_dir, verify)
        return 0

    if args.base_url:
        download_from_base_url(args.base_url, args.output_dir, verify)
        return 0

    print("Provide --source-dir, --base-url or --verify-only.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
