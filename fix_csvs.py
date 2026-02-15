"""
Recover and fix CSV files with encoding or malformed line issues.
Reads with utf-8, skips bad lines, and writes clean output.
"""

import os
from typing import Dict, List, Optional, Tuple

import pandas as pd

DEFAULT_ENCODINGS = ["utf-8", "utf-8-sig", "cp1252", "latin1"]


def _try_read_csv_strict(path: str) -> Optional[Tuple[pd.DataFrame, str]]:
    """Try encodings with normal read (no skip). Returns (df, enc) or None if all fail."""
    for enc in DEFAULT_ENCODINGS:
        try:
            df = pd.read_csv(path, encoding=enc, engine="python")
            print(f"Worked with encoding: {enc}")
            return df, enc
        except Exception:
            print(f"Failed with {enc}")
    return None


def _try_read_csv_with_skip(path: str) -> Tuple[pd.DataFrame, str]:
    """Try encodings with on_bad_lines='skip'. Use when strict read fails."""
    for enc in DEFAULT_ENCODINGS:
        try:
            df = pd.read_csv(path, encoding=enc, engine="python", on_bad_lines="skip")
            print(f"Fixed with encoding: {enc} (skipped bad lines)")
            return df, enc
        except Exception:
            print(f"Failed with {enc}")
    raise ValueError(f"Could not read {path} with any of {DEFAULT_ENCODINGS}")


def fix_csv(
    input_path: str,
    output_path: Optional[str] = None,
    overwrite: bool = False,
) -> pd.DataFrame:
    """
    Try strict reads first (utf-8, utf-8-sig, cp1252, latin1). If all fail, retry with
    on_bad_lines='skip' to recover. Writes a cleaned utf-8 version.

    Args:
        input_path: Path to the CSV file to fix.
        output_path: Where to write the cleaned CSV. If None, writes to
            'recovered_<filename>' in the same directory unless overwrite=True.
        overwrite: If True and output_path is None, overwrites the input file.

    Returns:
        The recovered DataFrame.
    """
    result = _try_read_csv_strict(input_path)
    if result is None:
        print("All strict reads failed, attempting fix (skipping bad lines)...")
        df, _ = _try_read_csv_with_skip(input_path)
    else:
        df, _ = result

    if output_path is None:
        if overwrite:
            output_path = input_path
        else:
            base, name = os.path.split(input_path)
            output_path = os.path.join(base, f"recovered_{name}")

    df.to_csv(output_path, index=False, encoding="utf-8")
    print(f"Recovered rows: {len(df)}")
    print(f"Saved as {output_path}")
    return df


def fix_all_csvs(
    directory: str = ".",
    patterns: Optional[List[str]] = None,
    overwrite: bool = False,
) -> Dict[str, pd.DataFrame]:
    """
    Fix all matching CSV files in a directory.

    Args:
        directory: Directory containing the CSV files.
        patterns: Filenames to fix (e.g. ["users_data_real.csv", "releases_data_real.csv"]).
            If None, uses the default project CSV names.
        overwrite: If True, overwrites the original files.

    Returns:
        Dict mapping filename -> recovered DataFrame.
    """
    if patterns is None:
        patterns = [
            "users_data_real.csv",
            "releases_data_real.csv",
            "artists_data_real.csv",
        ]

    results = {}
    for name in patterns:
        path = os.path.join(directory, name)
        if os.path.exists(path):
            print(f"\nProcessing {name}...")
            results[name] = fix_csv(path, overwrite=overwrite)
        else:
            print(f"Skipping {name} (not found)")

    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Fix CSV files with encoding issues.")
    parser.add_argument(
        "files",
        nargs="*",
        default=None,
        help="CSV files to fix (default: all project CSVs in current dir)",
    )
    parser.add_argument(
        "--overwrite",
        "-o",
        action="store_true",
        help="Overwrite original files instead of writing recovered_<name>",
    )
    args = parser.parse_args()

    if args.files:
        for f in args.files:
            print(f"\nProcessing {f}...")
            fix_csv(f, overwrite=args.overwrite)
    else:
        fix_all_csvs(".", overwrite=args.overwrite)
