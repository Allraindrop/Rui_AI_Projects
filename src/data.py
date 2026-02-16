from typing import Optional, Tuple
from datasets import load_dataset
from .utils import DatasetBundle

def _standardize(split, text_col: str, label_col: str):
    # keep only text/label columns
    if text_col != "text":
        split = split.rename_column(text_col, "text")
    if label_col != "label":
        split = split.rename_column(label_col, "label")
    keep_cols = {"text", "label"}
    drop_cols = [c for c in split.column_names if c not in keep_cols]
    if drop_cols:
        split = split.remove_columns(drop_cols)
    return split

def load_imdb(valid_size: int = 5000) -> DatasetBundle:
    ds = load_dataset("imdb")  # train/test provided
    train_full = _standardize(ds["train"], text_col="text", label_col="label")

    # create a validation split from train
    train_valid = train_full.train_test_split(test_size=valid_size, seed=42, shuffle=True)
    train = train_valid["train"]
    valid = train_valid["test"]

    test = _standardize(ds["test"], text_col="text", label_col="label")
    return DatasetBundle(name="imdb", train=train, valid=valid, test=test)

def load_sst2() -> DatasetBundle:
    ds = load_dataset("glue", "sst2")  # train/validation/test
    train = _standardize(ds["train"], text_col="sentence", label_col="label")
    valid = _standardize(ds["validation"], text_col="sentence", label_col="label")
    # glue sst2 test has no labels in some setups; keep it anyway
    test = _standardize(ds["test"], text_col="sentence", label_col="label") if "label" in ds["test"].column_names else ds["test"].rename_column("sentence", "text")
    return DatasetBundle(name="sst2", train=train, valid=valid, test=test)
