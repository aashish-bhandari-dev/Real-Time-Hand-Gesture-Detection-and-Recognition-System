"""Download the official MediaPipe Hand Landmarker task bundle."""

import sys
import urllib.request
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import ASSETS_DIR, MODEL_TASK_PATH, MODEL_TASK_URL


def download_model(force: bool = False) -> Path:
    """Download the official MediaPipe Hand Landmarker model task asset if not present."""
    ASSETS_DIR.mkdir(parents=True, exist_ok=True)

    if MODEL_TASK_PATH.exists() and not force:
        print(f"Model asset already exists at: {MODEL_TASK_PATH}")
        return MODEL_TASK_PATH

    print(f"Downloading MediaPipe Hand Landmarker model from:\n  {MODEL_TASK_URL}")
    print(f"Destination: {MODEL_TASK_PATH}")

    def report_progress(block_num: int, block_size: int, total_size: int) -> None:
        downloaded = block_num * block_size
        if total_size > 0:
            percent = min(100.0, downloaded * 100 / total_size)
            mb_downloaded = downloaded / (1024 * 1024)
            mb_total = total_size / (1024 * 1024)
            print(f"\rProgress: {percent:5.1f}% ({mb_downloaded:5.1f}MB / {mb_total:5.1f}MB)", end="")
        else:
            print(f"\rDownloaded: {downloaded / (1024 * 1024):5.1f}MB", end="")

    try:
        urllib.request.urlretrieve(MODEL_TASK_URL, MODEL_TASK_PATH, reporthook=report_progress)
        print("\nDownload complete successfully.")
    except Exception as e:
        if MODEL_TASK_PATH.exists():
            MODEL_TASK_PATH.unlink()
        print(f"\nFailed to download model asset: {e}", file=sys.stderr)
        raise

    return MODEL_TASK_PATH


if __name__ == "__main__":
    download_model()
