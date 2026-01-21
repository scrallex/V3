#!/usr/bin/env python3
"""CLI shim to render structural overlay plots for an STM state."""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Tuple

from scripts.plot_structural_window import generate_plots


def parse_tokens_from_directory(directory: Path) -> Tuple[datetime, datetime]:
    name = directory.name
    if "_" not in name:
        raise ValueError(f"Cannot infer start/stop from directory name: {name}")
    date_part, time_range = name.split("_", 1)
    if "-" not in time_range:
        raise ValueError(f"Cannot infer time range from directory name: {name}")
    start_token, stop_token = time_range.split("-", 1)
    def fmt(token: str) -> str:
        token = token.strip().replace("Z", "")
        if len(token) != 4:
            raise ValueError(f"Unexpected time token: {token}")
        return f"{token[:2]}:{token[2:]}:00"
    start_iso = f"{date_part}T{fmt(start_token)}"
    stop_iso = f"{date_part}T{fmt(stop_token)}"
    return datetime.fromisoformat(start_iso), datetime.fromisoformat(stop_iso)


def infer_config_path(state_path: Path) -> Path:
    stem = state_path.stem
    slug = stem.replace("mms_", "").removesuffix("_state")
    candidates = [
        state_path.with_name(f"router_config_{slug}.json"),
        Path("analysis") / f"router_config_{slug}.json",
        state_path.with_name("router_config.json"),
        Path("analysis") / "router_config.json",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError("Could not infer router config path. Pass --config explicitly.")


def infer_bits_path(state_settings: dict) -> Path:
    directory = Path(state_settings.get("directory", ""))
    if not directory.exists():
        raise FileNotFoundError(f"State directory not found: {directory}")
    candidates = sorted(directory.glob("*_bits.csv"))
    if not candidates:
        raise FileNotFoundError(f"No *_bits.csv found in {directory}")
    return candidates[0]


def infer_numeric_path(bits_path: Path) -> Path:
    slice_dir = bits_path.parent
    name = slice_dir.name
    if "_" not in name:
        raise ValueError(f"Cannot infer numeric path from directory name: {name}")
    date_part = name.split("_", 1)[0]
    numeric_dir = slice_dir.parents[1] / "csv" / date_part
    candidate = numeric_dir / "mms1_fgm.csv"
    if candidate.exists():
        return candidate
    csv_candidates = sorted(numeric_dir.glob("*.csv"))
    if not csv_candidates:
        raise FileNotFoundError(f"No numeric CSV found in {numeric_dir}")
    return csv_candidates[0]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render structural overlay plots for an STM state")
    parser.add_argument("--state", type=Path, required=True, help="State JSON path")
    parser.add_argument("--config", type=Path, help="Router config JSON (optional)")
    parser.add_argument("--bits", type=Path, help="Bit-feature CSV override")
    parser.add_argument("--numeric", type=Path, help="Numeric CSV override")
    parser.add_argument("--start", type=str, help="Start timestamp (ISO8601)")
    parser.add_argument("--stop", type=str, help="Stop timestamp (ISO8601)")
    parser.add_argument("--zoom-end", type=str, help="Zoom window end timestamp (ISO8601)")
    parser.add_argument("--zoom-minutes", type=int, default=10, help="Zoom window length in minutes")
    parser.add_argument("--out-dir", type=Path, default=Path("docs/plots"))
    parser.add_argument("--base-name", type=str, help="Filename prefix (defaults to state stem)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    state_path = args.state
    state_data = json.loads(state_path.read_text(encoding="utf-8"))

    config_path = args.config or infer_config_path(state_path)
    bits_path = args.bits or infer_bits_path(state_data.get("settings", {}))
    numeric_path = args.numeric or infer_numeric_path(bits_path)

    if args.start and args.stop:
        start = datetime.fromisoformat(args.start)
        stop = datetime.fromisoformat(args.stop)
    else:
        start, stop = parse_tokens_from_directory(bits_path.parent)

    zoom_end = datetime.fromisoformat(args.zoom_end) if args.zoom_end else start + (stop - start) * 2 / 3

    slug = args.base_name or state_path.stem.replace("_state", "")
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    out_prefix = out_dir / slug

    generate_plots(
        bits_path=bits_path,
        numeric_path=numeric_path,
        state_path=state_path,
        config_path=config_path,
        start=start,
        stop=stop,
        zoom_minutes=args.zoom_minutes,
        zoom_end=zoom_end,
        out_prefix=out_prefix,
    )
    print(f"Generated plots → {out_prefix}_overview.png, {out_prefix}_zoom.png")


if __name__ == "__main__":
    main()
