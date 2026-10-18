class ImmutableDict(dict):
    def __init__(self, *args, **kwargs):
        dict.update(self, dict(*args, **kwargs))

    def __setitem__(self, key, value):
        raise TypeError("ImmutableDict is frozen")

    def __delitem__(self, key):
        raise TypeError("ImmutableDict is frozen")

    def update(self, *args, **kwargs):
        raise TypeError("ImmutableDict is frozen")

    clear = pop = popitem = setdefault = update


def freeze_dict_attrs(obj, attrs) -> None:
    for key in attrs:
        value = getattr(obj, key)
        if isinstance(value, dict):
            object.__setattr__(obj, key, ImmutableDict(value))
