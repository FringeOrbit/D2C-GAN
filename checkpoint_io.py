"""Restricted checkpoint loading for the historical covariance release."""

from collections import OrderedDict
from collections.abc import Mapping
import pickle
from types import SimpleNamespace

import torch

from config import OptimizedConfig


class _RestrictedUnpickler(pickle.Unpickler):
    """Compatibility loader for PyTorch versions predating safe_globals."""

    def find_class(self, module, name):
        allowed = {
            ("config", "OptimizedConfig"): OptimizedConfig,
            ("collections", "OrderedDict"): OrderedDict,
            ("builtins", "set"): set,
            ("torch", "device"): torch.device,
        }
        for storage in ("FloatStorage", "DoubleStorage", "HalfStorage", "BFloat16Storage",
                        "LongStorage", "IntStorage", "ShortStorage", "CharStorage",
                        "ByteStorage", "BoolStorage"):
            allowed[("torch", storage)] = getattr(torch, storage)
        for rebuild in ("_rebuild_tensor", "_rebuild_tensor_v2", "_rebuild_parameter"):
            allowed[("torch._utils", rebuild)] = getattr(torch._utils, rebuild)
        if (module, name) not in allowed:
            raise pickle.UnpicklingError(f"Unsupported checkpoint global: {module}.{name}")
        return allowed[(module, name)]


def load_generator_state(path, map_location="cpu"):
    """Load tensors and the known config container without arbitrary globals.

    The saved config is metadata, not proof of the historical training loss.
    Modern PyTorch uses weights_only=True. Older versions use a small explicit
    global allowlist; neither path falls back to unrestricted pickle loading.
    """
    if hasattr(torch.serialization, "safe_globals"):
        with torch.serialization.safe_globals([OptimizedConfig]):
            checkpoint = torch.load(path, map_location=map_location, weights_only=True)
    else:
        restricted = SimpleNamespace(__name__="restricted_checkpoint_pickle",
                                     Unpickler=_RestrictedUnpickler)
        checkpoint = torch.load(path, map_location=map_location,
                                pickle_module=restricted, weights_only=False)
    if not isinstance(checkpoint, Mapping):
        raise TypeError("Checkpoint must be a state dictionary or mapping container")
    state = checkpoint.get("generator", checkpoint.get("model", checkpoint))
    if not isinstance(state, Mapping) or not state:
        raise TypeError("Generator state must be a nonempty mapping")
    if not all(isinstance(key, str) and torch.is_tensor(value) for key, value in state.items()):
        raise TypeError("Generator state must contain only named tensors")
    return {key.removeprefix("module."): value for key, value in state.items()}
