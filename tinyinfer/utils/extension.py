from functools import lru_cache


@lru_cache(maxsize=1)
def load_extension():
    try:
        from tinyinfer import _C
    except ImportError as exc:
        raise RuntimeError(
            "tinyinfer._C is not built; run: "
            "MAX_JOBS=4 python -m pip install -e . --no-build-isolation"
        ) from exc

    return _C