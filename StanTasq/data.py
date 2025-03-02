from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import prettytable
from pydantic import BaseModel
from typing import Any
from numpydantic import NDArray, Shape

from .calc_hash import EntityHash, calc_hash


def pretty_shape_dim(data: float | np.ndarray[float]) -> str:
    """
    Returns the number of dimensions of the shape.
    """
    if isinstance(data, float):
        return "1"
    shape = data.shape
    return "×".join(str(x) for x in shape)


class DataSet(BaseModel):
    """
    Class that holds the data for the Stan model. This is a Pydantic model.
    """

    data: dict[str, float | NDArray[Shape["*, ..."], Any]]  # noqa F722

    def __repr__(self) -> str:
        ans = "Data layout:"
        table = prettytable.PrettyTable()
        table.field_names = ["Name", "dimensions"]
        for name, data in self.data.items():
            table.add_row([name, pretty_shape_dim(data)])
        ans += str(table)
        return ans

    @property
    def data_hash(self) -> EntityHash:
        return calc_hash(self.data)

    @staticmethod
    def LoadData(data_file: str | Path) -> DataSet:
        """Loads data from a structured file. Right now, the supported format is only JSON and Pickle."""
        if isinstance(data_file, str):
            data_file = Path(data_file)

        assert isinstance(data_file, Path)
        assert data_file.exists()
        assert data_file.is_file()

        if data_file.suffix == ".json":
            json_data = data_file.read_text()
            data = json.loads(json_data)
        elif data_file.suffix == ".pkl":
            import pickle

            with data_file.open("rb") as f:
                data = pickle.load(f)
        else:
            raise ValueError("Unknown data file type")
        return DataSet(data=data)
