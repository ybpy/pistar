"""Keep tqdm bars readable when log lines rewrite the terminal."""


def sync_pbar_color(pbar) -> None:
    target = getattr(pbar, "tqdm", pbar)
    if hasattr(target, "colour"):
        target.colour = "cyan"


def update(n: int = 1) -> None:
    del n


def set_postfix(**kwargs) -> None:
    del kwargs


def set_postfix_str(text: str) -> None:
    del text


def set_description(text: str) -> None:
    del text


def close() -> None:
    return None
