#!/usr/bin/env python3
"""CLI shim to compute and plot lead-time foreground density."""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

from scripts.lead_time_density import compute_lead_summary, plot_lead_summary

from sep_text_manifold.cli_plots import infer_bits_path, infer_config_path, parse_tokens_from_directory


def parse_cli_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Lead-time foreground density summary")
    parser.add_argument("--state", type=Path, required=True, help="State JSON path")
    parser.add_argument("--config", type=Path, help="Router config JSON (optional)")
    parser.add_argument("--start", type=str, help="Start timestamp (ISO8601)")
    parser.add_argument("--stop", type=str, help="Stop timestamp (ISO8601)")
    parser.add_argument("--onset", type=str, required=True, help="Onset timestamp (ISO8601)")
    parser.add_argument("--bin", type=int, default=5, help="Bin size in minutes")
    parser.add_argument("--lookback", type=int, default=20, help="Minutes to look back from onset")
    parser.add_argument("--output", type=Path, help="Output JSON path (defaults to analysis/<slug>_leadtime.json)")
    parser.add_argument("--plot", type=Path, help="Optional plot path (defaults to docs/plots/<slug>_leadtime.png)")
    parser.add_argument("--out-dir", type=Path, default=Path("analysis"), help="Directory for JSON output")
    parser.add_argument("--plots-dir", type=Path, default=Path("docs/plots"), help="Directory for plot output")
    parser.add_argument("--base-name", type=str, help="File prefix base name")
    return parser.parse_args()


def main() -> None:
    args = parse_cli_args()
    state_path = args.state
    state_data = json.loads(state_path.read_text(encoding="utf-8"))
    config_path = args.config or infer_config_path(state_path)

    if args.start and args.stop:
        start = datetime.fromisoformat(args.start)
        stop = datetime.fromisoformat(args.stop)
    else:
        bits_path = infer_bits_path(state_data.get("settings", {}))
        start, stop = parse_tokens_from_directory(bits_path.parent)

    onset = datetime.fromisoformat(args.onset)

    slug = args.base_name or state_path.stem.replace("_state", "")
    output_json = args.output or (args.out_dir / f"{slug}_leadtime.json")
    plot_path = args.plot or (args.plots_dir / f"{slug}_leadtime.png")

    summary = compute_lead_summary(
        state_path=state_path,
        config_path=config_path,
        start=start,
        stop=stop,
        onset=onset,
        bin_minutes=args.bin,
        lookback_minutes=args.lookback,
    )
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    if plot_path:
        plot_lead_summary(summary, plot_path)
    print(f"Lead-time summary → {output_json}")
    if plot_path:
        print(f"Lead-time plot    → {plot_path}")


if __name__ == "__main__":
    main()
