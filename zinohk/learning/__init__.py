"""zinohk.learning — local learning rules"""
from zinohk.learning.contrastive import contrastive_update
from zinohk.learning.memory      import MemorySystem, FastMemory, SlowMemory
__all__ = ["contrastive_update", "MemorySystem",
           "FastMemory", "SlowMemory"]
