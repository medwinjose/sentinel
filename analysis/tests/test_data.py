"""Unit tests for sentinel_analysis.data modules."""

import json

from sentinel_analysis.data.fetch_binance_l2 import fetch_l2_data, generate_mock_l2_data
from sentinel_analysis.data.manifest import DataManifest
from sentinel_analysis.data.preprocess import preprocess_l2_data


def test_mock_l2_data_generation():
    depth, trades = generate_mock_l2_data("BTCUSDT")
    assert "bids" in depth
    assert "asks" in depth
    assert len(depth["bids"]) == 10
    assert len(depth["asks"]) == 10
    assert len(trades) == 10


def test_fetch_l2_data_dry_run(tmp_path):
    depth_file, trades_file, _depth_data, _trades_data = fetch_l2_data(
        symbol="BTCUSDT",
        raw_dir=tmp_path,
        dry_run=True,
    )

    assert depth_file.exists()
    assert trades_file.exists()
    assert depth_file.name.endswith("_depth.json")
    assert trades_file.name.endswith("_trades.json")

    with open(depth_file, "r", encoding="utf-8") as f:
        loaded_depth = json.load(f)
    assert "bids" in loaded_depth or "lastUpdateId" in loaded_depth


def test_preprocess_l2_data(tmp_path):
    depth_data, trades_data = generate_mock_l2_data("BTCUSDT")
    df, parquet_path, manifest_path = preprocess_l2_data(
        depth_data=depth_data,
        trades_data=trades_data,
        symbol="BTCUSDT",
        processed_dir=tmp_path,
        dry_run=True,
        seed=42,
    )

    assert parquet_path.exists()
    assert manifest_path.exists()
    assert manifest_path.name == "data_manifest.json"

    # Schema verification
    expected_cols = {"timestamp", "price", "size", "side", "event_type"}
    assert set(df.columns) == expected_cols

    # Types verification
    assert df["timestamp"].dtype == "float64"
    assert df["price"].dtype == "float64"
    assert df["size"].dtype == "float64"

    # Order verification (monotonically non-decreasing timestamp)
    assert df["timestamp"].is_monotonic_increasing

    # Manifest verification
    manifest = DataManifest.load(manifest_path)
    assert manifest.pair == "BTCUSDT"
    assert manifest.dry_run is True
    assert manifest.seed == 42
    assert manifest.row_counts["processed_events"] == len(df)
