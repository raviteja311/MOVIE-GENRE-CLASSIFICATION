# Data

`raw/` holds the three source files. Every line is one movie, with fields separated by ` ::: `.

| File | Format | Rows |
| --- | --- | ---: |
| `train_data.txt` | `ID ::: TITLE ::: GENRE ::: DESCRIPTION` | 54,214 |
| `test_data.txt` | `ID ::: TITLE ::: DESCRIPTION` | 54,200 |
| `test_data_solution.txt` | `ID ::: TITLE ::: GENRE ::: DESCRIPTION` | 54,200 |

There are 27 genres. The test files share the same `ID`s, so the solution file supplies labels for held-out evaluation.

To restore the files from another location:

```bash
movie-genre-download --source-dir <path-to-data>
```

Copies and downloads are checked against the SHA-256 checksums in `src/movie_genre/config.py` (`DATA_SHA256`), computed after converting CRLF line endings to LF. Check the files in place with `movie-genre-download --verify-only`.

Source: ftp://ftp.fu-berlin.de/pub/misc/movies/database/
