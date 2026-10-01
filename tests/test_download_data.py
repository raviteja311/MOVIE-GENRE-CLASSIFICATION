import pytest

from movie_genre.config import DATA_DIR, DATA_FILES, DATA_SHA256
from movie_genre.download_data import file_sha256, verify_file


def test_checksum_ignores_line_endings_and_detects_changes(tmp_path):
    lf, crlf = tmp_path / "lf.txt", tmp_path / "crlf.txt"
    lf.write_bytes(b"1 ::: A ::: drama ::: plot\n")
    crlf.write_bytes(b"1 ::: A ::: drama ::: plot\r\n")
    assert file_sha256(lf) == file_sha256(crlf)

    verify_file(lf, file_sha256(lf))
    with pytest.raises(ValueError, match="Checksum mismatch"):
        verify_file(lf, "0" * 64)


@pytest.mark.skipif(not all((DATA_DIR / name).exists() for name in DATA_FILES), reason="data files not available")
def test_committed_data_matches_checksums():
    for name in DATA_FILES:
        verify_file(DATA_DIR / name, DATA_SHA256[name])
