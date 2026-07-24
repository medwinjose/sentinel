"""Binance L2 Order Book Depth & Trade Data Acquisition Module."""

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import requests

BINANCE_API_BASE = "https://api.binance.com"


def get_default_raw_dir() -> Path:
    """Return default raw data directory path: sentinel/data/raw."""
    # sentinel_analysis/data/fetch_binance_l2.py -> sentinel/
    # parents: [0]=data, [1]=sentinel_analysis, [2]=analysis, [3]=sentinel
    repo_root = Path(__file__).resolve().parents[3]
    return repo_root / "data" / "raw"


def fetch_depth_snapshot(symbol: str, limit: int = 1000) -> dict[str, Any]:
    """Fetch order book depth snapshot from Binance REST API."""
    url = f"{BINANCE_API_BASE}/api/v3/depth"
    params = {"symbol": symbol.upper(), "limit": limit}
    response = requests.get(url, params=params, timeout=10)
    response.raise_for_status()
    return response.json()


def fetch_recent_trades(symbol: str, limit: int = 1000) -> list[dict[str, Any]]:
    """Fetch recent public trades from Binance REST API."""
    url = f"{BINANCE_API_BASE}/api/v3/trades"
    params = {"symbol": symbol.upper(), "limit": limit}
    response = requests.get(url, params=params, timeout=10)
    response.raise_for_status()
    return response.json()


def fetch_trades_paginated(
    symbol: str, target_count: int = 5000
) -> list[dict[str, Any]]:
    """Fetch target_count consecutive trade events from Binance using aggTrades pagination."""
    if target_count <= 1000:
        return fetch_recent_trades(symbol, limit=target_count)

    url = f"{BINANCE_API_BASE}/api/v3/aggTrades"
    latest_resp = requests.get(
        url, params={"symbol": symbol.upper(), "limit": 1000}, timeout=10
    )
    latest_resp.raise_for_status()
    latest_trades = latest_resp.json()

    if not latest_trades:
        return []

    max_a = latest_trades[-1]["a"]
    start_a = max(0, max_a - target_count + 1)
    all_trades = []

    current_a = start_a
    while len(all_trades) < target_count:
        fetch_len = min(1000, target_count - len(all_trades))
        resp = requests.get(
            url,
            params={"symbol": symbol.upper(), "fromId": current_a, "limit": fetch_len},
            timeout=10,
        )
        if resp.status_code != 200:
            break
        chunk = resp.json()
        if not chunk:
            break
        all_trades.extend(chunk)
        current_a = chunk[-1]["a"] + 1

    return all_trades[:target_count]


def generate_mock_l2_data(symbol: str) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Generate synthetic mock L2 depth + trade data for offline dry-run testing."""
    now_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
    mock_depth = {
        "lastUpdateId": 123456789,
        "bids": [
            [f"{65000 - i * 0.5:.2f}", f"{0.1 + i * 0.05:.4f}"] for i in range(10)
        ],
        "asks": [
            [f"{65000.5 + i * 0.5:.2f}", f"{0.1 + i * 0.05:.4f}"] for i in range(10)
        ],
    }
    mock_trades = [
        {
            "id": 1000 + i,
            "price": f"{65000 + (i % 2 - 0.5):.2f}",
            "qty": f"{0.05 + i * 0.01:.4f}",
            "time": now_ms - (10 - i) * 100,
            "isBuyerMaker": (i % 2 == 0),
            "isBestMatch": True,
        }
        for i in range(10)
    ]
    return mock_depth, mock_trades


def fetch_l2_data(
    symbol: str = "BTCUSDT",
    raw_dir: Path | None = None,
    dry_run: bool = False,
    limit: int = 1000,
    num_trades: int = 1000,
) -> tuple[Path, Path, dict[str, Any], list[dict[str, Any]]]:
    """Fetch depth & trade data for a symbol and save raw output locally.

    Returns:
        Tuple of (depth_raw_file, trades_raw_file, depth_data, trades_data)
    """
    target_dir = raw_dir if raw_dir else get_default_raw_dir()
    target_dir.mkdir(parents=True, exist_ok=True)

    fetch_limit = 10 if dry_run else limit

    if dry_run:
        try:
            depth_data = fetch_depth_snapshot(symbol, limit=fetch_limit)
            trades_data = fetch_recent_trades(symbol, limit=fetch_limit)
        except Exception as e:  # noqa: BLE001
            # Fallback to mock data if network is unavailable during dry-run
            print(
                f"[DRY-RUN] Network request failed ({e}), using synthetic L2 mock data."
            )
            depth_data, trades_data = generate_mock_l2_data(symbol)
    else:
        depth_data = fetch_depth_snapshot(symbol, limit=fetch_limit)
        if num_trades > 1000:
            trades_data = fetch_trades_paginated(symbol, target_count=num_trades)
        else:
            trades_data = fetch_recent_trades(symbol, limit=num_trades)

    timestamp_str = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    prefix = f"{symbol.lower()}_{timestamp_str}"
    if dry_run:
        prefix = f"{prefix}_dryrun"

    depth_file = target_dir / f"{prefix}_depth.json"
    trades_file = target_dir / f"{prefix}_trades.json"

    with open(depth_file, "w", encoding="utf-8") as f:
        json.dump(depth_data, f, indent=2)

    with open(trades_file, "w", encoding="utf-8") as f:
        json.dump(trades_data, f, indent=2)

    return depth_file, trades_file, depth_data, trades_data


def main():
    parser = argparse.ArgumentParser(description="Fetch Binance L2 depth & trade data.")
    parser.add_argument(
        "--symbol", type=str, default="BTCUSDT", help="Trading pair symbol"
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Perform dry-run fetch test"
    )
    parser.add_argument(
        "--num-trades", type=int, default=1000, help="Number of trade events to fetch"
    )
    parser.add_argument(
        "--out-dir", type=str, default=None, help="Directory for raw data output"
    )
    args = parser.parse_args()

    raw_dir = Path(args.out_dir) if args.out_dir else None
    depth_file, trades_file, depth_data, trades_data = fetch_l2_data(
        symbol=args.symbol,
        raw_dir=raw_dir,
        dry_run=args.dry_run,
        num_trades=args.num_trades,
    )

    print(f"Fetch completed successfully (dry_run={args.dry_run}):")
    print(f"  Raw Depth File : {depth_file}")
    print(f"  Raw Trades File: {trades_file}")
    if "bids" in depth_data:
        print(f"  Bids Count     : {len(depth_data['bids'])}")
        print(f"  Asks Count     : {len(depth_data['asks'])}")
    print(f"  Trades Count   : {len(trades_data)}")


if __name__ == "__main__":
    main()
