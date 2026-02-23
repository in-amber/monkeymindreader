#!/usr/bin/env python
"""
Packages the submission zip for the NSF HDR Neural Forecasting Challenge.

Creates submission.zip containing (flat, no subdirectory):
  model.py            — submission Model class
  architecture.py     — NeuralForecaster architecture
  model_beignet.pth   — trained Beignet checkpoint
  model_affi.pth      — trained Affi checkpoint

Usage:
    python make_submission.py

CAUTION: Do NOT zip the folder itself — only the file contents.
The submission platform (Codabench) will error with:
  ModuleNotFoundError: No module named 'model'
if the files are nested inside a directory in the zip.
"""

import os
import shutil
import zipfile

ROOT = os.path.dirname(__file__)
SUBMISSION_DIR = os.path.join(ROOT, 'submission')
CHECKPOINTS_DIR = os.path.join(ROOT, 'checkpoints')
OUTPUT_ZIP = os.path.join(ROOT, 'submission.zip')

# Files to include
SUBMISSION_FILES = [
    os.path.join(SUBMISSION_DIR, 'model.py'),
    os.path.join(SUBMISSION_DIR, 'architecture.py'),
]

CHECKPOINT_MAP = {
    os.path.join(CHECKPOINTS_DIR, 'model_beignet_mega.pth'): 'model_beignet.pth',
    os.path.join(CHECKPOINTS_DIR, 'model_affi_mega.pth'):    'model_affi.pth',
}


def make_submission():
    # Verify all required files exist
    missing = []
    for src in SUBMISSION_FILES:
        if not os.path.exists(src):
            missing.append(src)
    for src in CHECKPOINT_MAP:
        if not os.path.exists(src):
            missing.append(src)

    if missing:
        print("ERROR: Missing required files:")
        for f in missing:
            print(f"  {f}")
        print("\nMake sure training has completed and checkpoints exist.")
        return False

    # Build the zip
    with zipfile.ZipFile(OUTPUT_ZIP, 'w', zipfile.ZIP_DEFLATED) as zf:
        # Add submission source files (flat — just the filename, no directory)
        for src in SUBMISSION_FILES:
            arcname = os.path.basename(src)
            zf.write(src, arcname)
            print(f"  + {arcname}")

        # Add checkpoints (copy with submission-expected names)
        for src, dest_name in CHECKPOINT_MAP.items():
            zf.write(src, dest_name)
            src_size_mb = os.path.getsize(src) / 1e6
            print(f"  + {dest_name}  ({src_size_mb:.1f} MB)")

    zip_size_mb = os.path.getsize(OUTPUT_ZIP) / 1e6
    print(f"\nCreated: {OUTPUT_ZIP}  ({zip_size_mb:.1f} MB)")
    print("\nVerifying zip contents (must be flat — no subdirectories):")
    with zipfile.ZipFile(OUTPUT_ZIP, 'r') as zf:
        for name in zf.namelist():
            print(f"  {name}")

    return True


if __name__ == '__main__':
    print("Packaging submission...")
    ok = make_submission()
    if ok:
        print("\nDone. Submit submission.zip to Codabench.")
    else:
        print("\nFailed — fix missing files and try again.")
