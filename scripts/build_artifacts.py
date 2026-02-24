#!/usr/bin/env python3
"""
Wheel build orchestrator — reads artifacts.yaml and builds sequentially.

Usage:
    ./build_artifacts.py                          # build everything enabled
    ./build_artifacts.py --dry-run                # show what would be built
    ./build_artifacts.py --filter-package onnx*   # build matching packages
    ./build_artifacts.py --filter-image 25.11*    # build for matching images only
    ./build_artifacts.py --filter-image ubuntu*   # build for custom base only
    ./build_artifacts.py --package bitsandbytes   # build one specific package
"""

import argparse
import fnmatch
import os
import subprocess
import sys
import time
from pathlib import Path

try:
    import yaml
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "pyyaml", "-q"])
    import yaml


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR
WHEELS_DIR = PROJECT_DIR / "wheels"
DOCKERFILES_DIR = PROJECT_DIR / "dockerfiles"


def load_config(path: Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def detect_cuda_version(base_image: str) -> str:
    """Detect CUDA version from the base image via nvcc."""
    try:
        result = subprocess.run(
            ["docker", "run", "--rm", base_image, "nvcc", "--version"],
            capture_output=True, text=True, timeout=60,
        )
        for line in result.stdout.splitlines():
            if "release" in line:
                import re
                match = re.search(r"release (\d+\.\d+)", line)
                if match:
                    return "cu" + match.group(1).replace(".", "")
    except Exception:
        pass
    return "unknown"


def image_output_dir(img_config: dict) -> str:
    """Get the output directory name for a base image config."""
    if "output_dir" in img_config:
        return img_config["output_dir"]
    # Fallback: extract tag from URI
    tag = img_config["tag"]
    return tag.split(":")[-1] if ":" in tag else tag


def image_filter_name(img_config: dict) -> str:
    """Get the name to match against for --filter-image."""
    tag = img_config["tag"]
    return tag.split(":")[-1] if ":" in tag else tag


def build_wheel(base_image: str, package: str, version_arg: str, version: str,
                image_dir: Path) -> bool:
    """Build a single wheel. Returns True on success."""
    dockerfile = DOCKERFILES_DIR / package / "Dockerfile"
    if not dockerfile.exists():
        print(f"  ✗ No Dockerfile for {package} at {dockerfile}")
        return False

    cmd = [
        "docker", "build",
        "--build-arg", f"BASE_IMAGE={base_image}",
        "--build-arg", f"{version_arg}={version}",
        "--output", f"type=local,dest={image_dir}",
        "-f", str(dockerfile),
        str(DOCKERFILES_DIR / package),
    ]

    print(f"\n  $ docker build ... {package} @ {version}")
    start = time.time()
    result = subprocess.run(cmd, capture_output=False)
    elapsed = time.time() - start

    if result.returncode == 0:
        print(f"  ✓ {package} built in {elapsed:.0f}s → {image_dir}/")
        return True
    else:
        print(f"  ✗ {package} FAILED after {elapsed:.0f}s (exit {result.returncode})")
        return False


def main():
    parser = argparse.ArgumentParser(description="Build wheels from artifacts.yaml")
    parser.add_argument("--config", default=str(PROJECT_DIR / "artifacts.yaml"),
                        help="Path to artifacts.yaml")
    parser.add_argument("--dry-run", action="store_true",
                        help="Show build plan without building")
    parser.add_argument("--filter-image", default=None,
                        help="Glob filter for base image tags (e.g. '25.11*' or 'ubuntu*')")
    parser.add_argument("--filter-package", default=None,
                        help="Glob filter for package names (e.g. 'onnx*')")
    parser.add_argument("--package", default=None,
                        help="Build exactly this package (overrides enabled flag)")
    parser.add_argument("--include-disabled", action="store_true",
                        help="Also build packages marked enabled: false")
    args = parser.parse_args()

    config = load_config(Path(args.config))
    base_images = config.get("base_images", [])
    packages = config.get("packages", {})

    # ── Build plan ────────────────────────────────────────────────────
    plan = []
    for img in base_images:
        tag = img["tag"]
        filter_name = image_filter_name(img)
        out_dir = image_output_dir(img)

        if args.filter_image and not fnmatch.fnmatch(filter_name, args.filter_image):
            continue

        for pkg_name, pkg_cfg in packages.items():
            if not pkg_cfg.get("enabled", True):
                if not args.include_disabled and args.package != pkg_name:
                    continue

            if args.package and args.package != pkg_name:
                continue
            if args.filter_package and not fnmatch.fnmatch(pkg_name, args.filter_package):
                continue

            version_arg = pkg_cfg.get("version_arg",
                                     pkg_name.upper().replace("-", "_") + "_VERSION")

            plan.append({
                "base_image": tag,
                "output_dir": out_dir,
                "is_local": img.get("local", False),
                "package": pkg_name,
                "version_arg": version_arg,
                "version": pkg_cfg.get("version", "main"),
            })

    if not plan:
        print("Nothing to build. Check filters and artifacts.yaml.")
        sys.exit(0)

    # ── Print plan ────────────────────────────────────────────────────
    W = 88
    print(f"╔{'═' * (W - 2)}╗")
    print(f"║  {'DGX Spark Builder — Artifact Build Plan':<{W - 4}}║")
    print(f"╠{'═' * (W - 2)}╣")
    print(f"║  {'#':<4} {'Package':<20} {'Version':<16} {'Output Dir':<20} {'Local':<8} ║")
    print(f"║  {'─' * 4} {'─' * 20} {'─' * 16} {'─' * 20} {'─' * 8} ║")
    for i, job in enumerate(plan, 1):
        local = "✓" if job['is_local'] else ""
        line = f"  {i:<4} {job['package']:<20} {job['version']:<16} {job['output_dir']:<20} {local:<8} "
        print(f"║{line}║")
    print(f"╠{'═' * (W - 2)}╣")
    total_line = f"  Total: {len(plan)} build(s), sequential"
    print(f"║{total_line:<{W - 2}}║")
    print(f"╚{'═' * (W - 2)}╝")

    if args.dry_run:
        print("\n  --dry-run: exiting without building.")
        sys.exit(0)

    # ── Execute builds ────────────────────────────────────────────────
    results = []

    for job in plan:
        image_dir = WHEELS_DIR / job["output_dir"]
        image_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n{'━' * 52}")
        print(f"  [{len(results)+1}/{len(plan)}] {job['package']} @ {job['version']}")
        print(f"  Base: {job['base_image']} → {job['output_dir']}/")
        if job['is_local']:
            print(f"  ⚠ Local image — must be pre-built")
        print(f"{'━' * 52}")

        ok = build_wheel(
            base_image=job["base_image"],
            package=job["package"],
            version_arg=job["version_arg"],
            version=job["version"],
            image_dir=image_dir,
        )
        results.append((job["package"], job["output_dir"], ok))

    # ── Summary ───────────────────────────────────────────────────────
    print(f"\n{'━' * 52}")
    print("  Build Summary")
    print(f"{'━' * 52}")
    for pkg, out_dir, ok in results:
        status = "✓" if ok else "✗"
        print(f"  {status}  {pkg:20s}  [{out_dir}]")

    successes = sum(1 for _, _, ok in results if ok)
    failures = len(results) - successes
    print(f"\n  {successes} succeeded, {failures} failed")

    if failures > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
