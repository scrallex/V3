"""Utility to export OANDA trades for a time window into a Markdown report."""

from __future__ import annotations

import argparse
import logging
import os
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import requests

LOGGER = logging.getLogger("export_oanda_trades")

OANDA_LIVE = "https://api-fxtrade.oanda.com"
OANDA_PRACTICE = "https://api-fxpractice.oanda.com"


@dataclass
class TradeExit:
    time: datetime
    units: Decimal
    price: Optional[Decimal]
    reason: Optional[str]
    realized_pl: Decimal
    transaction_id: str


@dataclass
class TradeRecord:
    trade_id: str
    instrument: str
    direction: str
    initial_units: Decimal
    entry_time: datetime
    entry_price: Decimal
    entry_reason: Optional[str]
    entry_transaction: str
    stop_loss: Optional[Decimal] = None
    take_profit: Optional[Decimal] = None
    trailing_stop_distance: Optional[Decimal] = None
    exits: List[TradeExit] = field(default_factory=list)
    realized_pl: Decimal = Decimal("0")
    close_time: Optional[datetime] = None
    average_close_price: Optional[Decimal] = None
    state: str = "OPEN"
    current_units: Decimal = Decimal("0")

    def apply_exit(self, exit_event: TradeExit) -> None:
        self.exits.append(exit_event)
        self.realized_pl += exit_event.realized_pl
        self.current_units -= exit_event.units
        if self.current_units <= Decimal("0"):
            self.current_units = Decimal("0")
            self.state = "CLOSED"
            self.close_time = exit_event.time
            # Use realized weighted average if not already set.
            if exit_event.price is not None:
                self.average_close_price = exit_event.price


def parse_decimal(value: Any) -> Optional[Decimal]:
    if value in (None, "", "NA"):
        return None
    try:
        return Decimal(str(value))
    except (InvalidOperation, TypeError):
        return None


def parse_decimal_abs(value: Any) -> Decimal:
    parsed = parse_decimal(value)
    return abs(parsed) if parsed is not None else Decimal("0")


def parse_time(value: str) -> datetime:
    if value.endswith("Z"):
        value = value.replace("Z", "+00:00")
    return datetime.fromisoformat(value).astimezone(timezone.utc)


def format_time(dt: Optional[datetime]) -> str:
    if not dt:
        return "-"
    return dt.astimezone(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")


def format_decimal(value: Optional[Decimal], *, places: Optional[int] = None) -> str:
    if value is None:
        return "-"
    if places is not None:
        return f"{value:.{places}f}"
    normalized = value.normalize()
    text = f"{normalized:f}"
    if "." in text:
        text = text.rstrip("0").rstrip(".")
    return text if text else "0"


def chunked(iterable: Iterable[str], size: int) -> Iterable[List[str]]:
    batch: List[str] = []
    for item in iterable:
        batch.append(item)
        if len(batch) >= size:
            yield batch
            batch = []
    if batch:
        yield batch


def resolve_api_base() -> str:
    environment = os.getenv("OANDA_ENVIRONMENT", "live").lower()
    return OANDA_LIVE if environment == "live" else OANDA_PRACTICE


def build_session() -> requests.Session:
    api_key = os.getenv("OANDA_API_KEY")
    if not api_key:
        raise RuntimeError("OANDA_API_KEY is not set; ensure OANDA.env is loaded")
    session = requests.Session()
    session.headers.update(
        {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "Accept-Datetime-Format": "RFC3339",
        }
    )
    return session


def fetch_transaction_pages(
    session: requests.Session,
    account_id: str,
    base_url: str,
    start: datetime,
    end: datetime,
    *,
    page_size: int = 1000,
) -> List[Dict[str, Any]]:
    url = f"{base_url}/v3/accounts/{account_id}/transactions"
    params = {
        "from": start.isoformat().replace("+00:00", "Z"),
        "to": end.isoformat().replace("+00:00", "Z"),
        "pageSize": page_size,
    }
    response = session.get(url, params=params, timeout=30)
    response.raise_for_status()
    payload = response.json()
    pages = payload.get("pages", [])
    transactions: List[Dict[str, Any]] = []
    for page_url in pages:
        page_response = session.get(page_url, timeout=30)
        page_response.raise_for_status()
        page_data = page_response.json()
        transactions.extend(page_data.get("transactions", []) or [])
    return transactions


def fetch_trade_details(
    session: requests.Session, account_id: str, base_url: str, trade_ids: Iterable[str]
) -> Dict[str, Dict[str, Any]]:
    url = f"{base_url}/v3/accounts/{account_id}/trades"
    details: Dict[str, Dict[str, Any]] = {}
    for chunk in chunked(trade_ids, 50):
        params = {"ids": ",".join(chunk), "state": "ALL"}
        response = session.get(url, params=params, timeout=30)
        response.raise_for_status()
        for item in response.json().get("trades", []) or []:
            trade_id = item.get("id")
            if trade_id:
                details[trade_id] = item
    return details


def build_trade_records(transactions: List[Dict[str, Any]]) -> Dict[str, TradeRecord]:
    order_meta: Dict[str, Dict[str, Optional[Decimal]]] = {}
    trade_records: Dict[str, TradeRecord] = {}

    transactions.sort(key=lambda item: item.get("time", ""))

    for txn in transactions:
        txn_type = txn.get("type")
        txn_id = txn.get("id")
        if txn_type == "MARKET_ORDER" and txn_id:
            order_meta[txn_id] = {
                "take_profit": parse_decimal(txn.get("takeProfitOnFill", {}).get("price"))
                if isinstance(txn.get("takeProfitOnFill"), dict)
                else None,
                "stop_loss": parse_decimal(txn.get("stopLossOnFill", {}).get("price"))
                if isinstance(txn.get("stopLossOnFill"), dict)
                else None,
                "trailing_stop": parse_decimal(
                    txn.get("trailingStopLossOnFill", {}).get("distance")
                )
                if isinstance(txn.get("trailingStopLossOnFill"), dict)
                else None,
            }
            continue

        if txn_type != "ORDER_FILL":
            continue

        order_id = txn.get("orderID")
        meta = order_meta.get(order_id or "")
        trade_opened = txn.get("tradeOpened")
        if trade_opened:
            trade_id = trade_opened.get("tradeID")
            if not trade_id:
                continue
            units = parse_decimal(trade_opened.get("units")) or Decimal("0")
            direction = "LONG" if units >= 0 else "SHORT"
            entry_price = parse_decimal(txn.get("price")) or Decimal("0")
            record = trade_records.get(trade_id)
            if record is None:
                record = TradeRecord(
                    trade_id=trade_id,
                    instrument=txn.get("instrument", ""),
                    direction=direction,
                    initial_units=abs(units),
                    entry_time=parse_time(txn.get("time")),
                    entry_price=entry_price,
                    entry_reason=txn.get("reason"),
                    entry_transaction=txn.get("id", ""),
                    stop_loss=meta.get("stop_loss") if meta else None,
                    take_profit=meta.get("take_profit") if meta else None,
                    trailing_stop_distance=meta.get("trailing_stop") if meta else None,
                    current_units=abs(units),
                )
                trade_records[trade_id] = record
            else:
                record.initial_units += abs(units)
                record.current_units += abs(units)
        trades_closed = txn.get("tradesClosed") or []
        for closed in trades_closed:
            trade_id = closed.get("tradeID")
            if not trade_id:
                continue
            record = trade_records.get(trade_id)
            if not record:
                continue
            exit_units = parse_decimal_abs(closed.get("units"))
            exit_price = parse_decimal(closed.get("price")) or parse_decimal(txn.get("price"))
            realized_pl = parse_decimal(closed.get("realizedPL")) or Decimal("0")
            exit_event = TradeExit(
                time=parse_time(txn.get("time")),
                units=exit_units,
                price=exit_price,
                reason=txn.get("reason"),
                realized_pl=realized_pl,
                transaction_id=txn.get("id", ""),
            )
            record.apply_exit(exit_event)
        trade_reduced = txn.get("tradeReduced")
        if trade_reduced:
            trade_id = trade_reduced.get("tradeID")
            record = trade_records.get(trade_id)
            if record:
                exit_units = parse_decimal_abs(trade_reduced.get("units"))
                exit_price = parse_decimal(trade_reduced.get("price")) or parse_decimal(
                    txn.get("price")
                )
                realized_pl = parse_decimal(trade_reduced.get("realizedPL")) or Decimal("0")
                exit_event = TradeExit(
                    time=parse_time(txn.get("time")),
                    units=exit_units,
                    price=exit_price,
                    reason=txn.get("reason"),
                    realized_pl=realized_pl,
                    transaction_id=txn.get("id", ""),
                )
                record.apply_exit(exit_event)
    return trade_records


def merge_trade_details(records: Dict[str, TradeRecord], details: Dict[str, Dict[str, Any]]) -> None:
    for trade_id, trade in records.items():
        info = details.get(trade_id) or {}
        trade.state = info.get("state", trade.state)
        if info.get("closeTime"):
            trade.close_time = parse_time(info["closeTime"])
        if info.get("averageClosePrice"):
            trade.average_close_price = parse_decimal(info.get("averageClosePrice"))
        if info.get("realizedPL"):
            trade.realized_pl = parse_decimal(info.get("realizedPL")) or trade.realized_pl
        if not trade.stop_loss:
            stop_loss = info.get("stopLossOrder", {}) if isinstance(info.get("stopLossOrder"), dict) else {}
            trade.stop_loss = parse_decimal(stop_loss.get("price")) or trade.stop_loss
        if not trade.take_profit:
            take_profit = info.get("takeProfitOrder", {}) if isinstance(info.get("takeProfitOrder"), dict) else {}
            trade.take_profit = parse_decimal(take_profit.get("price")) or trade.take_profit


def render_report(
    records: Dict[str, TradeRecord],
    *,
    start: datetime,
    end: datetime,
    output_path: Path,
    account_id: str,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    trades_sorted = sorted(records.values(), key=lambda item: item.entry_time)
    total_realized = sum((trade.realized_pl for trade in trades_sorted), Decimal("0"))
    closed_trades = [t for t in trades_sorted if t.state == "CLOSED"]

    with output_path.open("w", encoding="utf-8") as handle:
        handle.write(
            f"# OANDA Trade Report ({start.date()} - {end.date()})\n\n"
        )
        handle.write(f"Account: `{account_id}`\n\n")
        handle.write("## Overview\n\n")
        handle.write(f"- Trades captured: {len(trades_sorted)}\n")
        handle.write(f"- Closed during window: {len(closed_trades)}\n")
        handle.write(f"- Aggregate realized P/L: {format_decimal(total_realized, places=2)}\n\n")

        for trade in trades_sorted:
            handle.write(
                f"## Trade {trade.trade_id} - {trade.instrument} ({trade.direction})\n\n"
            )
            handle.write(
                f"- Entry: {format_time(trade.entry_time)} at {format_decimal(trade.entry_price)}"
            )
            if trade.entry_reason:
                handle.write(f" (reason: {trade.entry_reason})\n")
            else:
                handle.write("\n")
            handle.write(f"- Initial units: {format_decimal(trade.initial_units)}\n")
            stop_line = (
                f"- Stop loss: {format_decimal(trade.stop_loss)} | Take profit: {format_decimal(trade.take_profit)}"
            )
            if trade.trailing_stop_distance:
                stop_line += (
                    f" | Trailing stop distance: {format_decimal(trade.trailing_stop_distance)}"
                )
            handle.write(f"{stop_line}\n")
            handle.write(f"- Status: {trade.state}\n")
            if trade.close_time:
                handle.write(
                    f"- Close time: {format_time(trade.close_time)} | Average close price: {format_decimal(trade.average_close_price)}\n"
                )
            handle.write(
                f"- Realized P/L: {format_decimal(trade.realized_pl, places=2)}\n\n"
            )
            if trade.exits:
                handle.write("| Exit Time | Units | Price | Reason | Realized P/L | Transaction |\n")
                handle.write("| --- | --- | --- | --- | --- | --- |\n")
                for exit_event in trade.exits:
                    handle.write(
                        "| {time} | {units} | {price} | {reason} | {pl} | {txn} |\n".format(
                            time=format_time(exit_event.time),
                            units=format_decimal(exit_event.units),
                            price=format_decimal(exit_event.price),
                            reason=exit_event.reason or "-",
                            pl=format_decimal(exit_event.realized_pl, places=2),
                            txn=exit_event.transaction_id or "-",
                        )
                    )
                handle.write("\n")
            else:
                handle.write("(No exit transactions recorded in this window.)\n\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export OANDA trades to Markdown")
    parser.add_argument("start", help="Inclusive start date (YYYY-MM-DD)")
    parser.add_argument("end", help="Inclusive end date (YYYY-MM-DD)")
    parser.add_argument(
        "--output",
        default=str(Path("output") / "oanda_trades_report.md"),
        help="Path for the generated Markdown report",
    )
    parser.add_argument(
        "--account-id",
        default=os.getenv("OANDA_ACCOUNT_ID"),
        help="Override the OANDA account id (defaults to environment)",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level))

    if not args.account_id:
        raise RuntimeError("OANDA_ACCOUNT_ID not provided; use --account-id or set env variable")

    start_date = datetime.strptime(args.start, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    end_date = datetime.strptime(args.end, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    end_inclusive = end_date + timedelta(days=1) - timedelta(microseconds=1)

    base_url = resolve_api_base()
    session = build_session()

    LOGGER.info(
        "Fetching transactions from %s to %s",
        start_date.isoformat(),
        end_inclusive.isoformat(),
    )
    transactions = fetch_transaction_pages(
        session, args.account_id, base_url, start_date, end_inclusive
    )
    if not transactions:
        LOGGER.warning("No transactions found for the requested window")

    records = build_trade_records(transactions)
    if not records:
        LOGGER.warning("No trade activity mapped in this window")

    details = fetch_trade_details(session, args.account_id, base_url, records.keys())
    merge_trade_details(records, details)

    output_path = Path(args.output)
    render_report(records, start=start_date, end=end_date, output_path=output_path, account_id=args.account_id)
    LOGGER.info("Report written to %s", output_path)


if __name__ == "__main__":
    main()
