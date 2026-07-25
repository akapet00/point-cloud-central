"""Download PCPNet and prepare reusable nearest-neighbor caches."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
from scipy.spatial import KDTree

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / "data" / "PCPNet"
RAW_DIR = DATA_DIR / "raw"
ARRAY_DIR = DATA_DIR / "arrays"
INPUT_DIR = DATA_DIR / "inputs"
CACHE_DIR = DATA_DIR / "cache"
JOB_DIR = DATA_DIR / "jobs"
DOWNLOAD_URL = "https://geometry.cs.ucl.ac.uk/projects/2018/pcpnet/pclouds.zip"
MAX_NEIGHBORS = 1_024
DEVELOPMENT_QUERIES_PER_SHAPE = 1_000
RANDOM_SEED = 42

TIER_SPLITS = {
    "development": "trainingset_vardensity_whitenoise.txt",
    "validation": "validationset_vardensity_whitenoise.txt",
    "test": "testset_all.txt",
}
CONDITIONS = ("clean", "low", "medium", "high", "stripe", "gradient")


def condition_from_name(name: str) -> str:
    """Return the PCPNet perturbation condition encoded in a shape name."""
    if name.endswith("_noise_white_1.00e-02"):
        return "low"
    if name.endswith("_noise_white_5.00e-02"):
        return "medium"
    if name.endswith("_noise_white_1.00e-01"):
        return "high"
    if name.endswith("_ddist_minmax_layers"):
        return "stripe"
    if name.endswith("_ddist_minmax"):
        return "gradient"
    return "clean"


def load_shape_names(tier: str) -> list[str]:
    """Load shape names for an evaluation tier."""
    split_path = RAW_DIR / TIER_SPLITS[tier]
    if not split_path.is_file():
        msg = f"Missing {split_path}. Run `uv run prepare.py download` first."
        raise FileNotFoundError(msg)
    return [
        line.strip() for line in split_path.read_text().splitlines() if line.strip()
    ]


def download_dataset() -> None:
    """Download and extract the official PCPNet point-cloud archive."""
    if (RAW_DIR / TIER_SPLITS["test"]).is_file():
        print(f"PCPNet is already available at {RAW_DIR}")
        return

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    archive_path = DATA_DIR / "pclouds.zip"

    if not archive_path.is_file():
        print(f"Downloading {DOWNLOAD_URL}")
        urllib.request.urlretrieve(DOWNLOAD_URL, archive_path)

    print("Checking archive")
    with zipfile.ZipFile(archive_path) as archive:
        corrupt_member = archive.testzip()
        if corrupt_member is not None:
            msg = f"Corrupt archive member: {corrupt_member}"
            raise zipfile.BadZipFile(msg)
        if any(
            Path(name).is_absolute() or ".." in Path(name).parts
            for name in archive.namelist()
        ):
            msg = "Archive contains an unsafe path"
            raise ValueError(msg)

        print(f"Extracting to {RAW_DIR}")
        RAW_DIR.mkdir(parents=True, exist_ok=True)
        archive.extractall(RAW_DIR)

    archive_path.unlink()
    verify_dataset()


def verify_dataset() -> None:
    """Check the official splits and files needed by the benchmark."""
    missing: list[Path] = []
    counts: dict[str, dict[str, int]] = {}

    for tier in TIER_SPLITS:
        names = load_shape_names(tier)
        tier_counts = {condition: 0 for condition in CONDITIONS}
        for name in names:
            tier_counts[condition_from_name(name)] += 1
            for suffix in (".xyz", ".normals", ".pidx"):
                path = RAW_DIR / f"{name}{suffix}"
                if not path.is_file():
                    missing.append(path)
        counts[tier] = tier_counts

    if missing:
        examples = "\n".join(str(path) for path in missing[:10])
        msg = f"PCPNet is incomplete; {len(missing)} files are missing:\n{examples}"
        raise FileNotFoundError(msg)

    print("PCPNet dataset verified")
    for tier, tier_counts in counts.items():
        summary = ", ".join(f"{key}={value}" for key, value in tier_counts.items())
        print(f"  {tier}: {summary}")


def _array_paths(name: str) -> tuple[Path, Path]:
    return ARRAY_DIR / f"{name}.points.npy", ARRAY_DIR / f"{name}.normals.npy"


def _prepare_arrays(name: str) -> tuple[np.ndarray, Path]:
    points_path, normals_path = _array_paths(name)
    ARRAY_DIR.mkdir(parents=True, exist_ok=True)

    if not points_path.is_file():
        points = np.loadtxt(RAW_DIR / f"{name}.xyz", dtype=np.float32)
        np.save(points_path, points)
    if not normals_path.is_file():
        normals = np.loadtxt(RAW_DIR / f"{name}.normals", dtype=np.float32)
        np.save(normals_path, normals)

    return np.load(points_path, mmap_mode="r"), normals_path


def _base_shape_name(name: str) -> str:
    """Remove a PCPNet perturbation suffix from a shape name."""
    for suffix in (
        "_noise_white_1.00e-02",
        "_noise_white_5.00e-02",
        "_noise_white_1.00e-01",
        "_ddist_minmax_layers",
        "_ddist_minmax",
    ):
        if name.endswith(suffix):
            return name.removesuffix(suffix)
    return name


def _development_queries(query_indices: np.ndarray, name: str) -> np.ndarray:
    count = min(DEVELOPMENT_QUERIES_PER_SHAPE, query_indices.size)
    base_name = _base_shape_name(name)
    digest = hashlib.sha256(f"{RANDOM_SEED}:{base_name}".encode()).digest()
    rng = np.random.default_rng(int.from_bytes(digest[:8], "little"))
    selected = rng.choice(query_indices.size, size=count, replace=False)
    return query_indices[np.sort(selected)]


def prepare_cache(tier: str, *, force: bool = False) -> None:
    """Prepare anonymous inputs and nearest-neighbor caches for one tier."""
    names = load_shape_names(tier)
    tier_dir = CACHE_DIR / tier
    input_dir = INPUT_DIR / tier
    tier_dir.mkdir(parents=True, exist_ok=True)
    input_dir.mkdir(parents=True, exist_ok=True)
    manifest_shapes: list[dict[str, object]] = []

    for index, name in enumerate(names, start=1):
        shape_id = f"shape-{index:03d}"
        cache_path = tier_dir / f"{shape_id}.npz"
        query_indices = np.loadtxt(RAW_DIR / f"{name}.pidx", dtype=np.int64, ndmin=1)
        if tier == "development":
            query_indices = _development_queries(query_indices, name)

        points, _ = _prepare_arrays(name)
        input_path = input_dir / f"{shape_id}.points.npy"
        if force or not input_path.is_file():
            np.save(input_path, points)
        if np.any(query_indices < 0) or np.any(query_indices >= points.shape[0]):
            msg = f"Invalid query index in {name}.pidx"
            raise ValueError(msg)

        cache_is_current = False
        if cache_path.is_file() and not force:
            try:
                with np.load(cache_path) as cache:
                    cache_is_current = (
                        cache["query_indices"].shape == query_indices.shape
                        and np.array_equal(cache["query_indices"], query_indices)
                        and cache["neighbor_indices"].shape
                        == (query_indices.size, MAX_NEIGHBORS)
                        and cache["neighbor_distances"].shape
                        == (query_indices.size, MAX_NEIGHBORS)
                    )
            except (KeyError, OSError, ValueError):
                cache_is_current = False

        if not cache_is_current:
            print(f"[{index}/{len(names)}] Querying neighbors for {name}")
            tree = KDTree(points)
            distances, neighbor_indices = tree.query(
                points[query_indices],
                k=MAX_NEIGHBORS,
                workers=-1,
            )
            temporary_path = cache_path.with_suffix(".tmp")
            with temporary_path.open("wb") as file:
                np.savez(
                    file,
                    query_indices=query_indices.astype(np.uint32),
                    neighbor_indices=neighbor_indices.astype(np.uint32),
                    neighbor_distances=distances.astype(np.float32),
                )
            temporary_path.replace(cache_path)
        else:
            print(f"[{index}/{len(names)}] Reusing {cache_path.name}")

        manifest_shapes.append(
            {
                "id": shape_id,
                "name": name,
                "condition": condition_from_name(name),
                "queries": int(query_indices.size),
            }
        )

    manifest = {
        "version": 1,
        "tier": tier,
        "split": TIER_SPLITS[tier],
        "max_neighbors": MAX_NEIGHBORS,
        "development_queries_per_shape": DEVELOPMENT_QUERIES_PER_SHAPE,
        "random_seed": RANDOM_SEED,
        "shapes": manifest_shapes,
    }
    manifest_path = DATA_DIR / f"{tier}-manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    (tier_dir / "manifest.json").unlink(missing_ok=True)
    input_manifest = {
        "version": 1,
        "tier": tier,
        "shape_ids": [shape["id"] for shape in manifest_shapes],
    }
    (input_dir / "inputs.json").write_text(json.dumps(input_manifest, indent=2) + "\n")
    print(f"Prepared {tier} cache at {tier_dir}")


def remove_generated_data() -> None:
    """Remove generated arrays and caches while retaining the raw dataset."""
    shutil.rmtree(ARRAY_DIR, ignore_errors=True)
    shutil.rmtree(INPUT_DIR, ignore_errors=True)
    shutil.rmtree(CACHE_DIR, ignore_errors=True)
    shutil.rmtree(JOB_DIR, ignore_errors=True)
    print("Removed generated PCPNet arrays and caches")


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("download", help="download and extract PCPNet")
    subparsers.add_parser("verify", help="verify the raw PCPNet files")

    cache_parser = subparsers.add_parser("cache", help="prepare an evaluation cache")
    cache_parser.add_argument("--tier", choices=TIER_SPLITS, required=True)
    cache_parser.add_argument("--force", action="store_true")
    cache_parser.add_argument(
        "--confirm-final-test",
        action="store_true",
        help="required to prepare the test tier",
    )
    subparsers.add_parser("clean", help="remove generated arrays and caches")
    return parser.parse_args()


def main() -> None:
    args = parse_arguments()
    if args.command == "download":
        download_dataset()
    elif args.command == "verify":
        verify_dataset()
    elif args.command == "clean":
        remove_generated_data()
    elif args.command == "cache":
        if args.tier == "test" and not args.confirm_final_test:
            msg = "Test cache preparation requires --confirm-final-test"
            raise SystemExit(msg)
        verify_dataset()
        prepare_cache(args.tier, force=args.force)


if __name__ == "__main__":
    main()
