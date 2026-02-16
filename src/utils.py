import json
import os
import random
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, Optional

import numpy as np

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)

def save_json(obj: Any, path: str) -> None:
    ensure_dir(os.path.dirname(path) or ".")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)

def load_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def now_str() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")

@dataclass
class DatasetBundle:
    name: str
    train: Any
    valid: Any
    test: Any
