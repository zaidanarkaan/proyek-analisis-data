"""Siapkan folder submission/ dan ZIP E-Commerce dengan manifest eksplisit."""

import argparse
import csv
import json
import shutil
import tempfile
from pathlib import Path
from urllib.parse import urlparse
from zipfile import ZIP_DEFLATED, ZipFile


ROOT = Path(__file__).resolve().parent
DATA_FILES = (
    "customers_dataset.csv",
    "orders_dataset.csv",
    "order_items_dataset.csv",
    "order_payments_dataset.csv",
    "order_reviews_dataset.csv",
    "products_dataset.csv",
    "product_category_name_translation.csv",
    "geolocation_dataset.csv",
)
PROJECT_FILES = (
    "notebook.ipynb",
    "README.md",
    "requirements.txt",
    "dashboard/dashboard.py",
    "dashboard/main_data.csv",
)


def prepare_data():
    """Salin CSV asli yang diperlukan ke data/ jika belum tersedia."""
    data_dir = ROOT / "data"
    missing = [name for name in DATA_FILES if not (data_dir / name).is_file()]
    source_dir = ROOT / "E-Commerce Public Dataset"
    for name in missing:
        if not (source_dir / name).is_file():
            raise FileNotFoundError(f"Dataset asli belum tersedia: data/{name}")
    if missing:
        data_dir.mkdir(exist_ok=True)
        for name in missing:
            shutil.copy2(source_dir / name, data_dir / name)
    print(f"Data siap: {len(DATA_FILES)} CSV E-Commerce di data/.")


def submission_manifest():
    files = [ROOT / name for name in PROJECT_FILES]
    files.extend(ROOT / "data" / name for name in DATA_FILES)
    for path in files:
        if not path.is_file():
            raise FileNotFoundError(f"Berkas submission belum tersedia: {path.relative_to(ROOT)}")

    dashboard_entries = {path.name for path in (ROOT / "dashboard").iterdir()}
    if dashboard_entries != {"dashboard.py", "main_data.csv"}:
        raise ValueError("Folder dashboard/ harus berisi tepat dashboard.py dan main_data.csv.")

    with (ROOT / "notebook.ipynb").open(encoding="utf-8") as notebook_file:
        notebook = json.load(notebook_file)
    code_cells = [cell for cell in notebook["cells"] if cell["cell_type"] == "code"]
    if not code_cells or any(
        cell.get("execution_count") is None
        or any(output.get("output_type") == "error" for output in cell.get("outputs", []))
        for cell in code_cells
    ):
        raise ValueError("Jalankan seluruh cell notebook.ipynb tanpa error sebelum membuat submission.")

    with (ROOT / "dashboard/main_data.csv").open(newline="", encoding="utf-8") as data_file:
        columns = set(next(csv.reader(data_file)))
    if not {"delivery_status", "estimated_delivery_days", "is_late", "delay_days"} <= columns:
        raise ValueError("Jalankan ulang notebook.ipynb untuk menghasilkan main_data.csv versi revisi.")

    url_path = ROOT / "url.txt"
    if url_path.is_file():
        urls = [line.strip() for line in url_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        if urls:
            parsed = urlparse(urls[0])
            if len(urls) != 1 or parsed.scheme not in {"http", "https"} or not parsed.hostname:
                raise ValueError("url.txt harus berisi satu URL dashboard publik.")
            if parsed.hostname not in {"localhost", "127.0.0.1", "::1"}:
                files.append(url_path)
    return files


def prepare_submission(files):
    """Kumpulkan berkas manifest ke struktur folder yang diminta reviewer."""
    submission_dir = ROOT / "submission"
    for path in files:
        destination = submission_dir / path.relative_to(ROOT)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, destination)
    print(f"Folder siap: {submission_dir.name}/ ({len(files)} berkas).")
    return submission_dir


def build_submission(output):
    prepare_data()
    files = submission_manifest()
    output = output.resolve()
    if not output.parent.is_dir():
        raise FileNotFoundError(f"Direktori output belum tersedia: {output.parent}")
    submission_dir = prepare_submission(files)
    with tempfile.NamedTemporaryFile(dir=output.parent, suffix=".zip", delete=False) as temporary:
        temporary_path = Path(temporary.name)
    try:
        with ZipFile(temporary_path, "w", compression=ZIP_DEFLATED, compresslevel=6) as archive:
            for path in files:
                relative_path = path.relative_to(ROOT)
                archive.write(
                    submission_dir / relative_path,
                    arcname=f"submission/{relative_path.as_posix()}",
                )
        temporary_path.replace(output)
    finally:
        temporary_path.unlink(missing_ok=True)
    print(f"ZIP siap: {output.name} ({output.stat().st_size / 1024**2:.2f} MiB, {len(files)} berkas).")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepare-data", action="store_true", help="Siapkan data/ tanpa membuat ZIP.")
    parser.add_argument("--output", type=Path, default=ROOT / "submission_ecommerce.zip")
    arguments = parser.parse_args()
    if arguments.prepare_data:
        prepare_data()
    else:
        build_submission(arguments.output)
