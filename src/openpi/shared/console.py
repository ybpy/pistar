"""Colored log strings used by the value-model scripts."""


def info(message: str) -> str:
    return f"\033[1;36m{message}\033[0m"


def ok(message: str) -> str:
    return f"\033[1;32m{message}\033[0m"


def warn(message: str) -> str:
    return f"\033[1;33m{message}\033[0m"


def error(message: str) -> str:
    return f"\033[1;31m{message}\033[0m"
