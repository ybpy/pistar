"""Shape annotations only. Checks stay off so this does not import TensorFlow.

Gemma writes `Int["B L"]`. Current jaxtyping wants `Int[Array, "B L"]`, so these
are plain markers.
"""


class _Marker:
    def __class_getitem__(cls, item):
        return cls


class Array(_Marker):
    pass


class PyTree(_Marker):
    pass


class Bool(_Marker):
    pass


class Float(_Marker):
    pass


class Int(_Marker):
    pass


class UInt8(_Marker):
    pass


class PRNGKey(_Marker):
    pass


def typechecked(fn):
    return fn
