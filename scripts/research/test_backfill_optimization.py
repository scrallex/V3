#!/usr/bin/env python3
"""Quick test to validate backfill optimizations."""

from __future__ import annotations

import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))


def test_pandas_available():
    """Verify pandas/numpy are installed."""
    try:
        import pandas as pd
        import numpy as np
        print(f"✓ pandas {pd.__version__} available")
        print(f"✓ numpy {np.__version__} available")
        return True
    except ImportError as e:
        print(f"✗ Missing dependency: {e}")
        print("Run: pip install pandas>=2.0.0 numpy>=1.24.0")
        return False


def test_csv_loading():
    """Test CSV loading performance."""
    from scripts.tools.backfill_gate_history import OfflineCandleStore
    
    candles_dir = REPO_ROOT / "docs" / "evidence" / "candles"
    if not candles_dir.exists():
        print(f"✗ Candles directory not found: {candles_dir}")
        return False
    
    # Find first CSV
    csv_files = list(candles_dir.glob("*.csv"))
    if not csv_files:
        print(f"✗ No CSV files found in {candles_dir}")
        return False
    
    test_file = csv_files[0]
    print(f"\nTesting CSV load: {test_file.name}")
    
    start = time.time()
    store = OfflineCandleStore(test_file)
    elapsed = time.time() - start
    
    print(f"✓ Loaded {len(store.timestamps):,} candles in {elapsed:.2f}s")
    
    if elapsed > 60:
        print(f"⚠ Load time seems slow ({elapsed:.1f}s). Expected <20s with pandas.")
        return False
    
    # Test slicing
    if len(store.timestamps) > 1000:
        start_ts = int(store.timestamps[100])
        end_ts = int(store.timestamps[200])
        slice_start = time.time()
        sliced = store.slice(start_ts, end_ts)
        slice_elapsed = time.time() - slice_start
        print(f"✓ Sliced {len(sliced)} candles in {slice_elapsed:.3f}s")
    
    return True


def test_backfill_single_instrument():
    """Test backfill for a short time range with one instrument."""
    from scripts.tools import backfill_gate_history
    
    candles_dir = REPO_ROOT / "docs" / "evidence" / "candles"
    if not candles_dir.exists():
        print(f"✗ Candles directory not found: {candles_dir}")
        return False
    
    output_dir = REPO_ROOT / "docs" / "evidence" / "test_output"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("\nTesting backfill with EUR_USD, 7 days...")
    
    args = [
        "--start", "2024-01-01T00:00:00Z",
        "--end", "2024-01-08T00:00:00Z",
        "--instruments", "EUR_USD",
        "--candles-dir", str(candles_dir),
        "--export-json", str(output_dir / "test_gates.jsonl"),
        "--export-roc-summary", str(output_dir / "test_roc.json"),
        "--isolation",
    ]
    
    start = time.time()
    result = backfill_gate_history.main(args)
    elapsed = time.time() - start
    
    if result != 0:
        print(f"✗ Backfill failed with code {result}")
        return False
    
    print(f"✓ Backfill completed in {elapsed:.2f}s")
    
    # Verify outputs exist
    gates_file = output_dir / "test_gates.jsonl"
    roc_file = output_dir / "test_roc.json"
    
    if not gates_file.exists():
        print(f"✗ Gates file not created: {gates_file}")
        return False
    
    if not roc_file.exists():
        print(f"✗ ROC summary not created: {roc_file}")
        return False
    
    # Check file sizes
    gates_size = gates_file.stat().st_size
    roc_size = roc_file.stat().st_size
    
    print(f"✓ Gates file: {gates_size:,} bytes")
    print(f"✓ ROC summary: {roc_size:,} bytes")
    
    if gates_size == 0:
        print("✗ Gates file is empty")
        return False
    
    return True


def main():
    print("=" * 60)
    print("Backfill Optimization Validation Test")
    print("=" * 60)
    
    tests = [
        ("Dependencies", test_pandas_available),
        ("CSV Loading", test_csv_loading),
        ("Backfill Pipeline", test_backfill_single_instrument),
    ]
    
    results = []
    for name, test_fn in tests:
        print(f"\n{'='*60}")
        print(f"Test: {name}")
        print("=" * 60)
        try:
            passed = test_fn()
            results.append((name, passed))
        except Exception as e:
            print(f"✗ Test failed with exception: {e}")
            import traceback
            traceback.print_exc()
            results.append((name, False))
    
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    
    for name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{status}: {name}")
    
    all_passed = all(passed for _, passed in results)
    
    if all_passed:
        print("\n✓ All tests passed! Pipeline is ready.")
        print("\nNext steps:")
        print("1. Install dependencies: pip install -r requirements.txt")
        print("2. Run single span test:")
        print("   python3 scripts/research/run_span_backfills.py \\")
        print("     --candles-dir docs/evidence/candles \\")
        print("     --span-id <span-id> \\")
        print("     --workers 4")
        return 0
    else:
        print("\n✗ Some tests failed. Fix issues before running full pipeline.")
        return 1


if __name__ == "__main__":
    sys.exit(main())