#!/usr/bin/env python3
"""Archive hyperparameter sweep artifacts/logs and 3-seed pretraining logs."""

import os
import shutil
import tarfile
import pathlib
import subprocess
import sys

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
ARCHIVE_DIR = REPO_ROOT / "archive" / "archive_20260915_pre_sweep"
ARCHIVE_TAR = REPO_ROOT / "archive" / "archive_20260915_pre_sweep.tar.gz"

def main():
    print("=" * 80)
    print("ARCHIVING PREVIOUS HYPERPARAMETER SWEEPS AND 3-SEED PRETRAINING LOGS")
    print("=" * 80)

    # 1. Create target directories
    ARCHIVE_DIR.mkdir(parents=True, exist_ok=True)
    archive_sweeps = ARCHIVE_DIR / "sweeps"
    archive_3seeds = ARCHIVE_DIR / "3seed_runs"
    archive_artifacts = ARCHIVE_DIR / "artifacts"
    
    archive_sweeps.mkdir(parents=True, exist_ok=True)
    archive_3seeds.mkdir(parents=True, exist_ok=True)
    archive_artifacts.mkdir(parents=True, exist_ok=True)

    # 2. Archive sweep logs from runs/sweeps
    runs_sweeps = REPO_ROOT / "runs" / "sweeps"
    if runs_sweeps.exists():
        target = archive_sweeps / "runs_sweeps"
        if target.exists():
            shutil.rmtree(target)
        print(f"Moving {runs_sweeps} -> {target}")
        shutil.move(str(runs_sweeps), str(target))

    # 3. Archive sweep artifacts
    for art_name in ["sweeps", "muon_sweep"]:
        art_dir = REPO_ROOT / "artifacts" / art_name
        if art_dir.exists():
            target = archive_artifacts / art_name
            if target.exists():
                shutil.rmtree(target)
            print(f"Copying {art_dir} -> {target}")
            shutil.copytree(str(art_dir), str(target))

    if (REPO_ROOT / "artifacts" / "sweep_summary.md").exists():
        shutil.copy(str(REPO_ROOT / "artifacts" / "sweep_summary.md"), str(archive_artifacts / "sweep_summary.md"))
    if (REPO_ROOT / "runs" / "sweep_summary.json").exists():
        shutil.move(str(REPO_ROOT / "runs" / "sweep_summary.json"), str(archive_sweeps / "sweep_summary.json"))

    # 4. Archive 3-seed pretraining runs from runs/
    runs_dir = REPO_ROOT / "runs"
    for item in list(runs_dir.iterdir()):
        if item.is_dir() and (item.name.startswith("125m_") or item.name.startswith("test_")):
            target = archive_3seeds / item.name
            if target.exists():
                shutil.rmtree(target)
            print(f"Moving {item} -> {target}")
            shutil.move(str(item), str(target))

    # 5. Archive pretraining artifacts
    for art_name in ["pretraining_2.5b", "production_3b", "phase6"]:
        art_dir = REPO_ROOT / "artifacts" / art_name
        if art_dir.exists():
            target = archive_artifacts / art_name
            if target.exists():
                shutil.rmtree(target)
            print(f"Copying {art_dir} -> {target}")
            shutil.copytree(str(art_dir), str(target))

    # 6. Create compressed archive
    print(f"\nCompressing archive to {ARCHIVE_TAR}...")
    with tarfile.open(ARCHIVE_TAR, "w:gz") as tar:
        tar.add(ARCHIVE_DIR, arcname=ARCHIVE_DIR.name)
    print(f"Created compressed archive: {ARCHIVE_TAR} ({ARCHIVE_TAR.stat().st_size / 1024:.1f} KB)")

    # 7. Write README inside archive
    readme_path = ARCHIVE_DIR / "README.md"
    with open(readme_path, "w") as f:
        f.write("# Archive of Previous Sweeps and 3-Seed Pretraining Runs\n\n")
        f.write("Archived on: 2026-09-15\n")
        f.write("Contents:\n")
        f.write("- `sweeps/`: All run logs and metrics from previous hyperparameter sweeps\n")
        f.write("- `3seed_runs/`: Metrics and summaries from 125M pretraining runs (CauchyLift and AdamW)\n")
        f.write("- `artifacts/`: Previous summary reports, JSON metrics, and sweep evaluations\n")
        f.write("- Note: Model checkpoint weights (.pt) were safely removed to conserve disk storage.\n")

    print("\n[SUCCESS] Archival completed successfully.")

if __name__ == "__main__":
    main()
