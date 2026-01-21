"""Backtest simulator package reusing live trading primitives."""

from .backtest_simulator import BacktestSimulator, SimulationParams, SimulationResult
from .signal_deriver import derive_signals

__all__ = [
    "BacktestSimulator",
    "SimulationParams",
    "SimulationResult",
    "derive_signals",
]
