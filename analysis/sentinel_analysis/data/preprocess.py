"""Deterministic L2 market data preprocessing and manifest logging."""

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from sentinel_analysis.data.manifest import DataManifest


def get_default_processed_dir() -> Path:
    """Return default processed data directory path: sentinel/data/processed."""
    repo_root = Path(__file__).resolve().parents[3]
    return repo_root / "data" / "processed"


def preprocess_l2_data(
    depth_data: dict[str, Any],
    trades_data: list[dict[str, Any]],
    symbol: str = "BTCUSDT",
    processed_dir: Path | None = None,
    version: str = "v1",
    dry_run: bool = False,
    seed: int | None = None,
) -> tuple[pd.DataFrame, Path, Path]:
    """Preprocess raw depth snapshot and trades JSON data into a clean event stream Parquet file.

    Schema:
        - timestamp (float64): relative seconds from start of dataset
        - price (float64): level price
        - size (float64): order/trade volume
        - side (str): 'bid' or 'ask'
        - event_type (str): 'limit_add', 'limit_cancel', 'trade'

    Returns:
        Tuple of (df_events, parquet_path, manifest_path)
    """
    events = []

    # 1. Parse Trade Events
    min_trade_time = float("inf")
    max_trade_time = float("-inf")

    for trade in trades_data:
        t_ms = float(trade.get("time", 0))
        if t_ms > 0:
            min_trade_time = min(min_trade_time, t_ms)
            max_trade_time = max(max_trade_time, t_ms)

        price = float(trade.get("price", 0.0))
        qty = float(trade.get("qty", trade.get("quantity", 0.0)))
        # isBuyerMaker True -> trade executed at bid (seller aggressive) -> side 'ask' / 'sell'
        is_buyer_maker = trade.get("isBuyerMaker", False)
        side = "ask" if is_buyer_maker else "bid"

        events.append(
            {
                "raw_time_ms": t_ms,
                "price": price,
                "size": qty,
                "side": side,
                "event_type": "trade",
            }
        )

    base_time_ms = min_trade_time if min_trade_time != float("inf") else 0.0

    # 2. Parse Depth Snapshot Events (Bids & Asks)
    bids = depth_data.get("bids", [])
    asks = depth_data.get("asks", [])

    for b in bids:
        p, q = float(b[0]), float(b[1])
        events.append(
            {
                "raw_time_ms": base_time_ms,
                "price": p,
                "size": q,
                "side": "bid",
                "event_type": "limit_add",
            }
        )

    for a in asks:
        p, q = float(a[0]), float(a[1])
        events.append(
            {
                "raw_time_ms": base_time_ms,
                "price": p,
                "size": q,
                "side": "ask",
                "event_type": "limit_add",
            }
        )

    if not events:
        df = pd.DataFrame(columns=["timestamp", "price", "size", "side", "event_type"])
    else:
        df_raw = pd.DataFrame(events)
        df_raw.sort_values(
            by=["raw_time_ms", "event_type"], ascending=[True, True], inplace=True
        )

        # Convert raw timestamps to relative seconds from base_time_ms
        df = pd.DataFrame()
        df["timestamp"] = (df_raw["raw_time_ms"] - base_time_ms) / 1000.0
        df["price"] = df_raw["price"].astype(np.float64)
        df["size"] = df_raw["size"].astype(np.float64)
        df["side"] = df_raw["side"].astype(str)
        df["event_type"] = df_raw["event_type"].astype(str)
        df.reset_index(drop=True, inplace=True)

    # Output directory handling
    target_dir = processed_dir if processed_dir else get_default_processed_dir()
    target_dir.mkdir(parents=True, exist_ok=True)

    file_prefix = f"{symbol.lower()}_events_{version}"
    if dry_run:
        file_prefix = f"{file_prefix}_dryrun"
    parquet_filename = f"{file_prefix}.parquet"
    parquet_path = target_dir / parquet_filename

    # Save processed dataframe as Parquet
    df.to_parquet(parquet_path, index=False)

    # Timestamps to ISO
    if base_time_ms > 0:
        start_iso = datetime.fromtimestamp(
            base_time_ms / 1000.0, timezone.utc
        ).isoformat()
    else:
        start_iso = datetime.now(timezone.utc).isoformat()

    if max_trade_time > float("-inf") and max_trade_time > 0:
        end_iso = datetime.fromtimestamp(
            max_trade_time / 1000.0, timezone.utc
        ).isoformat()
    else:
        end_iso = start_iso

    manifest = DataManifest(
        pair=symbol,
        start_time=start_iso,
        end_time=end_iso,
        source_endpoints={
            "depth": "https://api.binance.com/api/v3/depth",
            "trades": "https://api.binance.com/api/v3/trades",
        },
        row_counts={
            "raw_depth_bids": len(bids),
            "raw_depth_asks": len(asks),
            "raw_trades": len(trades_data),
            "processed_events": len(df),
        },
        processed_filename=parquet_filename,
        dry_run=dry_run,
        seed=seed,
    )

    manifest_path = manifest.save(target_dir, processed_file_path=parquet_path)

    return df, parquet_path, manifest_path


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Preprocess raw L2 depth & trade JSON files into Parquet."
    )
    parser.add_argument("--depth-file", type=str, help="Path to raw depth JSON file")
    parser.add_argument("--trades-file", type=str, help="Path to raw trades JSON file")
    parser.add_argument(
        "--symbol", type=str, default="BTCUSDT", help="Trading pair symbol"
    )
    parser.add_argument("--dry-run", action="store_true", help="Mark output as dry-run")
    parser.add_argument("--seed", type=int, default=None, help="Optional RNG seed")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[3]
    raw_dir = repo_root / "data" / "raw"

    if args.depth_file and args.trades_file:
        depth_path = Path(args.depth_file)
        trades_path = Path(args.trades_file)
    else:
        # Pick latest raw files in raw_dir
        depth_files = sorted(raw_dir.glob("*_depth.json"))
        trades_files = sorted(raw_dir.glob("*_trades.json"))
        if not depth_files or not trades_files:
            raise FileNotFoundError(
                "No raw depth/trades files found in sentinel/data/raw. Run fetch_binance_l2 first."
            )
        depth_path = depth_files[-1]
        trades_path = trades_files[-1]

    with open(depth_path, "r", encoding="utf-8") as f:
        depth_data = json.load(f)

    with open(trades_path, "r", encoding="utf-8") as f:
        trades_data = json.load(f)

    df, parquet_path, manifest_path = preprocess_l2_data(
        depth_data=depth_data,
        trades_data=trades_data,
        symbol=args.symbol,
        dry_run=args.dry_run,
        seed=args.seed,
    )

    print("Preprocessing completed:")
    print(f"  Processed Parquet : {parquet_path}")
    print(f"  Manifest JSON     : {manifest_path}")
    print(f"  Total Events      : {len(df)}")


if __name__ == "__main__":
    main()
