import os


def trace_enabled(name: str) -> bool:
    return os.getenv(name, "0") == "1"

