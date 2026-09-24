"""Device detection and logging utilities."""
import json
import logging
from pathlib import Path

import torch


def get_device(prefer: str = "auto") -> torch.device:
    if prefer == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    if prefer == "auto" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def describe_device() -> str:
    if torch.cuda.is_available():
        return f"CUDA: {torch.cuda.get_device_name(0)}"
    return f"CPU ({torch.get_num_threads()} threads)"


def get_logger(name="hybrid", logfile: str | None = None) -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        h = logging.StreamHandler()
        h.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
        logger.addHandler(h)
        if logfile:
            Path(logfile).parent.mkdir(parents=True, exist_ok=True)
            fh = logging.FileHandler(logfile)
            fh.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
            logger.addHandler(fh)
    return logger


def append_jsonl(path: str, record: dict):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as f:
        f.write(json.dumps(record) + "\n")
