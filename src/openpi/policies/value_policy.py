import dataclasses
import io

import numpy as np

from openpi import transforms


def _get_by_path(data: dict, path: str):
    # Direct key match first.
    if path in data:
        return data[path], True

    def _traverse(parts: list[str]):
        cur = data
        for part in parts:
            if not isinstance(cur, dict) or part not in cur:
                return None, False
            cur = cur[part]
        return cur, True

    if "/" in path:
        val, ok = _traverse(path.split("/"))
        if ok:
            return val, True
    if "." in path:
        val, ok = _traverse(path.split("."))
        if ok:
            return val, True
    return None, False


def _get_first(data: dict, candidates: list[str], *, required: bool = False, name: str = "value"):
    for key in candidates:
        val, ok = _get_by_path(data, key)
        if ok:
            return val
    if required:
        raise KeyError(f"Missing {name}. Tried keys: {candidates}. Available top-level keys: {list(data.keys())}")
    return None


def _parse_image(image) -> np.ndarray:
    if image is None:
        raise ValueError("Image is None")

    if isinstance(image, dict) and "bytes" in image:
        image = image["bytes"]

    if isinstance(image, dict) and "path" in image and image["path"] is not None:
        image = image["path"]

    if isinstance(image, (bytes, bytearray)):
        from PIL import Image

        image = Image.open(io.BytesIO(image)).convert("RGB")
        return np.asarray(image)

    if isinstance(image, str):
        from PIL import Image

        image = Image.open(image).convert("RGB")
        return np.asarray(image)

    try:
        import torch
    except Exception:
        torch = None

    if torch is not None and isinstance(image, torch.Tensor):
        image = image.detach().cpu().numpy()

    image = np.asarray(image)
    if np.issubdtype(image.dtype, np.floating):
        image = (255 * image).astype(np.uint8)
    if image.shape[0] == 3:
        image = np.transpose(image, (1, 2, 0))
    return image


def _normalize_scalar_label(value, *, name: str) -> np.float32:
    array = np.asarray(value, dtype=np.float32)
    if array.ndim == 0:
        return np.float32(array)
    if array.size == 1:
        return np.float32(array.reshape(()))
    raise ValueError(f"{name} must be scalar-like, got shape {array.shape}")


@dataclasses.dataclass(frozen=True)
class ValueInputs(transforms.DataTransformFn):
    """Map 1-3 independent camera streams to the value model's fixed per-batch schema."""

    camera_names: tuple[str, ...] = ("head", "left")

    def __call__(self, data: dict) -> dict:
        from openpi.shared.value_cameras import CAMERA_MODEL_KEYS, image_is_present, normalize_camera_names
        camera_names = normalize_camera_names(self.camera_names)
        aliases = {
            "head": ("image", "cam_high", "base_0_rgb"),
            "left": ("wrist_image", "wrist_image_left", "cam_left_wrist", "wrist_0_rgb", "left_wrist_0_rgb"),
            "right": ("right_wrist_image", "wrist_image_right", "cam_right_wrist", "right_wrist_0_rgb"),
        }
        raw = {}
        for name in camera_names:
            candidates = []
            for alias in aliases[name]:
                candidates.extend((alias, f"observation/{alias}", f"observation.{alias}",
                                   f"observation/images/{alias}", f"observation.images.{alias}",
                                   f"images/{alias}", f"images.{alias}", f"image/{alias}", f"image.{alias}"))
            for candidate in candidates:
                value, found = _get_by_path(data, candidate)
                if found and image_is_present(value):
                    raw[name] = value
                    break
        # Retain the former right-as-wrist fallback only for the legacy two-view layout.
        if camera_names == ("head", "left") and "left" not in raw:
            for alias in aliases["right"]:
                for candidate in (alias, f"observation/{alias}", f"observation.{alias}",
                                  f"observation/images/{alias}", f"observation.images.{alias}",
                                  f"images/{alias}", f"images.{alias}"):
                    value, found = _get_by_path(data, candidate)
                    if found and image_is_present(value):
                        raw["left"] = value
                        break
                if "left" in raw:
                    break
        if not raw:
            raise KeyError(f"No images for selected value cameras {camera_names}; keys={list(data)}")
        # Preserve the legacy wrist-only fallback when head was requested.
        if "head" in camera_names and "head" not in raw:
            first = next(iter(raw))
            raw["head"] = raw.pop(first)
        parsed = {name: _parse_image(value) for name, value in raw.items()}
        template = next(iter(parsed.values()))
        images = {CAMERA_MODEL_KEYS[name]: parsed[name] if name in parsed else np.zeros_like(template) for name in camera_names}
        masks = {CAMERA_MODEL_KEYS[name]: np.bool_(name in parsed) for name in camera_names}

        state = _get_first(
            data,
            [
                "observation/state",
                "observation.state",
                "state",
            ],
            required=False,
            name="state",
        )
        if state is None:
            state = np.zeros((1,), dtype=np.float32)

        prompt = _get_first(
            data,
            [
                "prompt",
                "task",
                "observation/task",
                "observation.task",
            ],
            required=False,
            name="prompt",
        )

        value = _get_first(
            data,
            [
                "value",
                "reward",
                "return",
            ],
            required=True,
            name="value",
        )

        inputs = {
            "state": state,
            "image": images,
            "image_mask": masks,
        }

        if "actions" in data:
            inputs["actions"] = data["actions"]
        if prompt is not None:
            inputs["prompt"] = prompt
        if value is not None:
            inputs["value"] = _normalize_scalar_label(value, name="value")

        return inputs
