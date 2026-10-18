from flax import linen as nn


class Identity(nn.Module):
    """Pass-through used by Gemma attention. Same role as kd.nn.Identity."""

    @nn.compact
    def __call__(self, inputs, *args, **kwargs):
        del args, kwargs
        return inputs
