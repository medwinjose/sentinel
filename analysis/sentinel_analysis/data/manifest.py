"""Data provenance logging utility for SENTINEL."""

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def compute_file_sha256(file_path: Path) -> str:
    """Compute SHA256 checksum of a file."""
    sha256_hash = hashlib.sha256()
    with open(file_path, "rb") as f:
        for byte_block in iter(lambda: f.read(65536), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()


class DataManifest:
    """Class representing data provenance metadata for processed L2 market datasets."""

    def __init__(
        self,
        pair: str,
        start_time: str,
        end_time: str,
        source_endpoints: dict[str, str],
        row_counts: dict[str, int],
        processed_filename: str,
        dry_run: bool = False,
        seed: int | None = None,
    ):
        self.pair = pair
        self.start_time = start_time
        self.end_time = end_time
        self.source_endpoints = source_endpoints
        self.row_counts = row_counts
        self.processed_filename = processed_filename
        self.dry_run = dry_run
        self.seed = seed
        self.created_at = datetime.now(timezone.utc).isoformat()

    def to_dict(self, processed_file_path: Path | None = None) -> dict[str, Any]:
        data = {
            "pair": self.pair,
            "time_range": {
                "start": self.start_time,
                "end": self.end_time,
            },
            "source_endpoints": self.source_endpoints,
            "row_counts": self.row_counts,
            "processed_filename": self.processed_filename,
            "dry_run": self.dry_run,
            "seed": self.seed,
            "created_at": self.created_at,
        }
        if processed_file_path and processed_file_path.exists():
            data["sha256"] = compute_file_sha256(processed_file_path)
        return data

    def save(self, output_dir: Path, processed_file_path: Path | None = None) -> Path:
        """Save manifest to data_manifest.json in target output directory."""
        output_dir.mkdir(parents=True, exist_ok=True)
        manifest_path = output_dir / "data_manifest.json"
        manifest_dict = self.to_dict(processed_file_path)
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest_dict, f, indent=2)
        return manifest_path

    @classmethod
    def load(cls, manifest_path: Path) -> "DataManifest":
        """Load DataManifest from JSON file."""
        with open(manifest_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        manifest = cls(
            pair=data["pair"],
            start_time=data["time_range"]["start"],
            end_time=data["time_range"]["end"],
            source_endpoints=data["source_endpoints"],
            row_counts=data["row_counts"],
            processed_filename=data["processed_filename"],
            dry_run=data.get("dry_run", False),
            seed=data.get("seed"),
        )
        manifest.created_at = data.get("created_at", manifest.created_at)
        return manifest
