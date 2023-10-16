from .compressor import *

compress_registry = {
    "uniform": UniformQuantizer,
    "topk": Topk,
    "binaryconversion": BinaryConversion,
    "qsgd": QsgdQuantizer,
    "signSGD": SignSGDCompressor
}