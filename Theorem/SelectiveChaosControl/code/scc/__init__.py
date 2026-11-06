"""Selective Chaos Control (SCC) for LLM agents."""
from .decompose import Dimension, Decomposer, SQL_DIMENSIONS, decompose_sql
from .sensor import ChaosSensor, ftle_from_series, implied_error_rate, pairwise_divergence
from .controller import ControllerBank, DimensionController
from .horizon import HorizonManager, predictability_horizon
from .master import MasterController

__all__ = [
    "Dimension", "Decomposer", "SQL_DIMENSIONS", "decompose_sql",
    "ChaosSensor", "ftle_from_series", "implied_error_rate", "pairwise_divergence",
    "ControllerBank", "DimensionController", "HorizonManager", "predictability_horizon",
    "MasterController",
]
