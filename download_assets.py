import os
import sys
import zipfile
import urllib.request
from pathlib import Path

RELEASE_URL = "https://github.com/ShreyaB002/Drishti-/releases/download/v1.0.0/drishti_assets_v1.0.0.zip"
ROOT = Path(__file__).resolve().parent
ZIP_TARGET = ROOT / "drishti_assets_v1.0.0.zip"

REQUIRED_FILES = [
    ROOT / "yolo11n.pt",
    ROOT / "anpr_engine" / "anpr" / "models" / "plate_detector.pt",
    ROOT / "test" / "virtual_fence.mp4",
    ROOT / "test" / "vehicle_detection.mp4",
    ROOT / "test" / "human_deetection.mp4",
    ROOT / "test" / "suspicious_activity.mp4",
    ROOT / "test" / "Automatic Number Plate Recognition (ANPR) _ Vehicle Number Plate Recognition (1).mp4",
]


def reporthook(block_num, block_size, total_size):
    downloaded = block_num * block_size
    if total_size > 0:
        percent = min(100.0, downloaded * 100 / total_size)
        down_mb = downloaded / (1024 * 1024)
        total_mb = total_size / (1024 * 1024)
        bar_len = 30
        filled_len = int(bar_len * percent / 100)
        bar = "=" * filled_len + ">" + " " * (bar_len - filled_len - 1) if filled_len < bar_len else "=" * bar_len
        sys.stdout.write(f"\r  [{bar}] {percent:5.1f}% ({down_mb:.1f} / {total_mb:.1f} MB)")
        sys.stdout.flush()


def main():
    print("=" * 65)
    print("  Drishti Asset Downloader (Model Weights & Sample Feeds)")
    print("=" * 65)

    missing = [f for f in REQUIRED_FILES if not f.exists()]
    if not missing:
        print("\nAll required model weights and test video feeds are already present!")
        print("You are ready to run the dashboard:")
        print("  python -m uvicorn fastapi_dashboard:app --host 0.0.0.0 --port 8000\n")
        return

    print(f"\nMissing {len(missing)} asset(s). Downloading package from GitHub Release v1.0.0...")
    print(f"Source: {RELEASE_URL}\n")

    try:
        urllib.request.urlretrieve(RELEASE_URL, ZIP_TARGET, reporthook=reporthook)
        print("\n\nDownload completed successfully! Extracting files...")

        with zipfile.ZipFile(ZIP_TARGET, "r") as zf:
            zf.extractall(ROOT)

        if ZIP_TARGET.exists():
            ZIP_TARGET.unlink()

        print("\nAll model weights and video feeds extracted successfully:")
        print("  [OK] anpr_engine/anpr/models/plate_detector.pt")
        print("  [OK] yolo11n.pt")
        print("  [OK] test/virtual_fence.mp4")
        print("  [OK] test/vehicle_detection.mp4")
        print("  [OK] test/human_deetection.mp4")
        print("  [OK] test/suspicious_activity.mp4")
        print("  [OK] test/Automatic Number Plate Recognition...mp4")
        print("\nSetup complete! You can now start Drishti:")
        print("  python -m uvicorn fastapi_dashboard:app --host 0.0.0.0 --port 8000\n")

    except Exception as exc:
        print(f"\n[Error] Failed to download or extract assets: {exc}")
        print(f"You can manually download the zip from:\n  {RELEASE_URL}\nand extract it into this folder.")
        sys.exit(1)


if __name__ == "__main__":
    main()
