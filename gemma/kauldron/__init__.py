"""Small stand-in for the Gemma import of kauldron.

The value-model path only needs type aliases, an identity layer, and the
checkpoint sharding helper. The full kauldron package also pulls TensorFlow.
"""

__version__ = "0"


def __getattr__(name: str):
    raise AttributeError('Please use "from kauldron import kd".')
