"""zinohk.encoding — text and spike encoding"""
from zinohk.encoding.spike import SpikeEncoder, SpikeDecoder, Spike
from zinohk.encoding.text  import TFIDFEncoder, tokenise, clean
__all__ = ["SpikeEncoder", "SpikeDecoder", "Spike",
           "TFIDFEncoder", "tokenise", "clean"]
