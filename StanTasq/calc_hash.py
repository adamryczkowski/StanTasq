from typing import Any
import numpy as np
import inspect
import hashlib
import struct


def calc_hash(value: Any) -> str:
    sha256 = hashlib.sha256()
    add_thing_to_hash(sha256, value)
    return sha256.hexdigest()


def calc_inthash(value: Any) -> int:
    sha256 = hashlib.sha256()
    add_thing_to_hash(sha256, value)
    return int.from_bytes(sha256.digest(), byteorder="big")


def add_thing_to_hash(sha256, value: Any):
    if isinstance(value, str):
        sha256.update(value.encode())
    elif isinstance(value, dict):
        key_list = list(value.keys())
        key_list.sort()
        for key in key_list:
            add_thing_to_hash(sha256, key)
            add_thing_to_hash(sha256, value[key])
    elif isinstance(value, set):
        for item in sorted(list(value)):
            add_thing_to_hash(sha256, item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            add_thing_to_hash(sha256, item)
    elif isinstance(value, np.ndarray):
        add_thing_to_hash(sha256, value.tolist())
    elif isinstance(value, float):
        sha256.update(struct.pack("d", value))
    elif isinstance(value, int):
        sha256.update(struct.pack("q", value))
    elif (
        inspect.ismodule(value)
        or inspect.isclass(value)
        or inspect.ismethod(value)
        or inspect.isfunction(value)
        or inspect.iscode(value)
    ):
        if value.__module__ == "builtins":
            sha256.update(value.__name__.encode())
        else:
            sha256.update(inspect.getsource(value).encode())
    else:
        try:
            sha256.update(value)
        except Exception as e:
            raise ValueError(f"Cannot hash value {value} of type {type(value)}") from e


def make_dict_serializable_in_place(d: dict) -> dict:
    """Turns all numpy arrays in the dictionary into lists"""
    for key in d:
        if isinstance(d[key], dict):
            d[key] = make_dict_serializable_in_place(d[key])
        elif isinstance(d[key], np.ndarray):
            d[key] = d[key].tolist()
    return d


def create_dict_serializable_copy(d: dict) -> dict:
    """Turns all numpy arrays in the dictionary into lists"""
    ans = {}
    for key in d:
        if isinstance(d[key], dict):
            ans[key] = create_dict_serializable_copy(d[key])
        elif isinstance(d[key], np.ndarray):
            ans[key] = d[key].tolist()
        elif isinstance(d[key], list):
            ans[key] = [create_dict_serializable_copy(item) for item in d[key]]
        elif isinstance(d[key], (float, int)):
            ans[key] = d[key]
        else:
            raise ValueError(f"Cannot serialize value {d[key]} of type {type(d[key])}")
    return ans