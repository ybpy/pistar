REQUIRED = "__KEY_REQUIRED__"


class Key:
    """Marker default for Gemma input key fields."""


class Path(tuple):
    @classmethod
    def from_jax_path(cls, jax_path):
        return cls(jax_path)

    def __str__(self):
        return ".".join(str(part) for part in self)
