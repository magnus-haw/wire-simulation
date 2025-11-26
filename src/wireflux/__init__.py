"""
wireflux: Adaptive wire-filament simulation engine with Biot–Savart electromagnetics.
"""

from .core.engine import MultiWireEngine
from .core.state import State

from .models.wires import Wire
from .models.newwires import NewWire

from .physics.biot_savart import biot_savart
from .physics.forces import JxB_force

__all__ = [
    "MultiWireEngine",
    "State",
    "Wire",
    "NewWire",
    "biot_savart",
    "JxB_force",
]
