import csv
from pathlib import Path
from typing import Any, Dict, List, Union


def export_to_csv(data: List[Dict[str, Any]], file_path: Union[str, Path]) -> None:
    file_path = Path(file_path)

    if not data:
        raise ValueError("No data to export")

    keys = data[0].keys()

    with file_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(data)


def export_to_tsv(data: List[Dict[str, Any]], file_path: Union[str, Path]) -> None:
    file_path = Path(file_path)

    if not data:
        raise ValueError("No data to export")

    keys = data[0].keys()

    with file_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys, delimiter="\t")
        writer.writeheader()
        writer.writerows(data)
