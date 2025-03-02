from __future__ import annotations

import datetime as dt
import math
from abc import ABC, abstractmethod
from enum import Enum
from typing import Optional, Iterator

# import numpy.typing as npt
import humanize
import numpy as np
import prettytable
from ValueWithError import ValueWithError, VectorOfValues
from numpydantic import NDArray, Shape
from pydantic import BaseModel, model_validator

from .ifaces import StanErrorType, StanResultEngine

ColumnsSelectorType = list[int] | int | list[str] | str | None


class Covariances(BaseModel):
    matrix: NDArray[Shape["*, *"], float]  # noqa F722

    @model_validator(mode="after")
    def validate_model(self):
        assert isinstance(self.matrix, np.ndarray)
        assert len(self.matrix.shape) == 2
        shape = self.matrix.shape
        assert shape[0] == shape[1]

    @property
    def covariance_shape(self) -> NDArray[Shape["len, len"], int]:
        return self.matrix.shape


class ParameterNames(BaseModel):
    """Class that holds names to all parameters in the Stan model.
    There are four representations of the parameters:
    * parameter - the name of the parameter. It may be multidimensional, i.e. it can hold multiple individual items.
    * flatlabel - the label of the paraemter. It is the parameter name with the shape index appended.
    * flatindex - the index of the parameter in the flat array of parameters. It is a global index of the parameter.
    * shapeindex - the index of the parameter in the multidimensional array of parameters. It is relative to the specific parameter.
    The trick is that the parameters can be multidimensional, so we need to
    know the shape of each parameter.
    """

    # noinspection PyDataclass
    parameter_shapes: dict[
        str, tuple[int, ...]
    ] = {}  # Order matters, as it determines the flatindices
    _flatindex_to_parameter_name: list[str]  # Translates index to parameter name
    _parameter_sizes: dict[
        str, tuple[int, int]
    ]  # Translates parameter name to its range in the flat array, in the format (index, number of elements)
    _flatlabel_to_flatindex: dict[str, int]  # Translates flatlabel to flatindex

    @model_validator(mode="after")
    def validate_model(self):
        assert len(self._parameter_sizes) == len(self.parameter_indices)

    def get_parameter_from_index(self, index: int) -> str:
        return self._flatindex_to_parameter_name[index]

    def __init__(self, parameter_shapes: dict[str, tuple[int, ...]]):
        self.parameter_shapes = parameter_shapes
        self._make_parameter_indices()
        super().__init__(
            parameter_shapes=parameter_shapes
        )  # For validation and conformity with BaseModel

    def _make_parameter_indices(self):
        total_scalar_count = 0
        for par_name, par_shape in self.parameter_shapes.items():
            if par_shape == () or (len(par_shape) == 1 and par_shape[0] == 1):
                scalar_count = 1
            else:
                scalar_count = math.prod(par_shape)
            self._parameter_sizes[par_name] = (total_scalar_count, scalar_count)
            total_scalar_count += scalar_count

        self._flatindex_to_parameter_name = ["" for _ in range(total_scalar_count)]
        self._flatlabel_to_flatindex = {}
        par_index = -1
        par_size = 1
        for par_name in self.parameter_shapes.keys():
            old_par_index, old_par_size = par_index, par_size
            par_index, par_size = self._parameter_sizes[par_name]
            assert par_index == old_par_index + old_par_size + 1
            for i in range(par_index, par_index + par_size):
                self._flatindex_to_parameter_name[i] = par_name
                self._flatlabel_to_flatindex[
                    f"{par_name}{self.pretty_shapeindex(self.flatindex_to_shapeindex(par_name, i))}"
                ] = i

        assert par_index + par_size == total_scalar_count

    @property
    def scalar_count(self) -> int:
        return len(self._flatindex_to_parameter_name)

    @property
    def parameter_names(self) -> list[str]:
        return list(self.parameter_shapes.keys())

    def get_flatlabels_for_parameter_name(self, par_name: str) -> Iterator[str]:
        for flatindex in range(self.scalar_count):
            shapeindex = self.flatindex_to_shapeindex(par_name, flatindex)
            yield f"{par_name}{self.pretty_shapeindex(shapeindex)}"

    @property
    def labels_for_all_scalar_elements(self) -> Iterator[str]:
        for par in self.parameter_names:
            yield from self.get_flatlabels_for_parameter_name(par)

    def is_parameter_scalar(self, par_name: str) -> bool:
        return len(self.parameter_shapes[par_name]) == 0 or (
            len(self.parameter_shapes[par_name]) == 1
            and self.parameter_shapes[par_name][0] == 1
        )

    def get_parameter_shape(self, par_name: str) -> list[int]:
        return list(self.parameter_shapes[par_name])

    def flatindex_to_shapeindex(self, par_name: str, flat_index: int) -> list[int]:
        shape = self.parameter_shapes[par_name]
        ans = []
        for i in range(len(shape) - 1, -1, -1):
            ans.append(flat_index % shape[i])
            flat_index //= shape[i]
        return ans[::-1]

    def shapeindex_to_flatindex(self, par_name, shape_index: list[int]) -> int:
        shape = self.parameter_shapes[par_name]
        ans = 0
        for i in range(len(shape)):
            ans *= shape[i]
            ans += shape_index[i]
        return ans

    def get_flatindex_from_flatlabel(self, flatlabel: str) -> int:
        return self._flatlabel_to_flatindex[flatlabel]

    def get_flatlabel_from_flatindex(self, flatindex: int) -> str:
        parameter_name = self._flatindex_to_parameter_name[flatindex]
        return f"{parameter_name}{self.pretty_shapeindex(self.flatindex_to_shapeindex(self._flatindex_to_parameter_name[flatindex], flatindex))}"

    @staticmethod
    def pretty_shapeindex(shape_index: list[int]) -> str:
        return "[" + ",".join(str(i) for i in shape_index) + "]"

    @property
    def flatnames(self) -> Iterator[str]:
        # noinspection PyTypeChecker
        return self._flatlabel_to_flatindex.keys()

    def get_flatindices_from_parameter_name(self, par_name: str) -> Iterator[int]:
        par_start, par_size = self._parameter_sizes[par_name]
        # noinspection PyTypeChecker
        return range(par_start, par_start + par_size)

    def get_flatindices_from_parnames(
        self, par_names: ColumnsSelectorType
    ) -> Iterator[int]:
        if par_names is None:
            # noinspection PyTypeChecker
            return range(self.scalar_count)
        if isinstance(par_names, str):
            par_names = [par_names]
        if isinstance(par_names, int):
            par_names = [par_names]
        assert isinstance(par_names, list)
        for par_name in par_names:
            if isinstance(par_name, str):
                try:
                    flat_index = self.get_flatindex_from_flatlabel(par_name)
                    yield flat_index
                except KeyError:
                    yield from self.get_flatindices_from_parameter_name(par_name)
            else:
                assert 0 <= par_name < self.scalar_count
                yield par_name

    def get_parameter_from_flatindex(self, flatindex: int) -> str:
        return self._flatindex_to_parameter_name[flatindex]


class ResultClassType(Enum):
    """
    Class used to discriminate between different types of results in order to provide clean deserialization with Pydantic.
    """

    MainEffects = "MainEffects"
    Covariances = "Covariances"
    Samples = "Samples"
    Error = "Error"
    Raw = "Raw"


class ResultErrorBase(BaseModel):
    error_type: StanErrorType
    inference_engine: StanResultEngine
    inference_runtime: dt.timedelta
    compilation_time: dt.timedelta
    error_message: Optional[str] = None
    model_name: Optional[str] = None
    worker_tag: str
    # noinspection PyDataclass
    warnings: list[str] = []

    def __init__(
        self,
        error_type: StanErrorType,
        inference_engine: StanResultEngine,
        inference_runtime: dt.timedelta,
        compilation_time: dt.timedelta,
        worker_tag: str,
        error_message: Optional[str] | dict[str, str] = None,
        model_name: Optional[str] = None,
        warnings: list[str] = None,
        **kwargs,
    ):
        if isinstance(error_message, dict):
            err = []
            for key, message in error_message.items():
                if message != "":
                    err.append(f"{key}: {message}")
            if len(err) == 0:
                error_message = None
            else:
                error_message = "\n".join(err)
        super().__init__(
            error_type=error_type,
            inference_engine=inference_engine,
            inference_runtime=inference_runtime,
            compilation_time=compilation_time,
            error_message=error_message,
            model_name=model_name,
            worker_tag=worker_tag,
            warnings=warnings,
            **kwargs,
        )

    def is_error(self) -> bool:
        return self.error_type != StanErrorType.NO_ERROR

    def formatted_runtime(self) -> str:
        if self.inference_runtime is None:
            return "Run time: not available"
        else:
            return f"Run taken: {humanize.precisedelta(self.inference_runtime)}"

    def formatted_compile_time(self) -> str:
        if self.inference_runtime is None:
            return "Model compilation time: not available"
        else:
            return f"Model compilation time: {humanize.precisedelta(self.inference_runtime)}"

    def __repr__(self):
        if self.model_name is not None:
            ans = f"{self.model_name}, "
        else:
            ans = ""
        ans += f"{self.inference_engine} running for {self.formatted_runtime():}\n"
        if self.is_error():
            ans += f"{repr(self.error_type)}: {self.error_message}"

        warnings = self.warnings
        if len(warnings) > 0:
            ans += "\nWarnings:\n"
            for warning in warnings:
                ans += warning + "\n"
        return ans


class I_ResultMainEffects(ABC):
    @abstractmethod
    def sample_count(self, flatindex: int) -> float | int: ...

    @abstractmethod
    def get_parameter_estimate_flat(self, flatindex: int) -> ValueWithError: ...

    @abstractmethod
    def get_parameter_mu(
        self, par_names: ColumnsSelectorType = None
    ) -> NDArray[Shape["*, ..."], float]: ...  # noqa F722

    @abstractmethod
    def get_parameter_sigma(
        self, par_names: ColumnsSelectorType = None
    ) -> NDArray[Shape["*, ..."], float]: ...  # noqa F722


class I_ResultCovariances(I_ResultMainEffects):
    @abstractmethod
    @property
    def covariance_matrix(self) -> tuple[Covariances, ParameterNames]: ...


class I_ResultSamples(I_ResultCovariances):
    @abstractmethod
    def get_raw_samples(self, par_names: ColumnsSelectorType = None) -> np.ndarray: ...

    @abstractmethod
    def get_special_raw_samples(
        self,
    ) -> Optional[np.ndarray]: ...  # Makes sense only for MCMC

    @abstractmethod
    def get_estimate_with_samples(self, flatindex: int) -> VectorOfValues: ...


class ResultBase(ResultErrorBase):
    parameter_names: ParameterNames

    def __init__(
        self,
        parameter_names: ParameterNames,
        error_type: StanErrorType,
        inference_engine: StanResultEngine,
        inference_runtime: dt.timedelta,
        compilation_time: dt.timedelta,
        worker_tag: str,
        error_message: Optional[str] = None,
        model_name: Optional[str] = None,
        warnings: list[str] = None,
        **kwargs,
    ):
        super().__init__(
            error_type=error_type,
            inference_engine=inference_engine,
            inference_runtime=inference_runtime,
            compilation_time=compilation_time,
            error_message=error_message,
            model_name=model_name,
            worker_tag=worker_tag,
            warnings=warnings,
            parameter_names=parameter_names,
            **kwargs,
        )

    def get_parameter_estimates(
        self, par_names: ColumnsSelectorType = None
    ) -> list[tuple[str, ValueWithError]]:
        flat_indices = self.parameter_names.get_flatindices_from_parnames(par_names)
        return [
            (
                self.parameter_names.get_parameter_from_flatindex(i),
                self.get_parameter_estimate_flat(i),
            )
            for i in flat_indices
        ]

    def pretty_cov_matrix(
        self, user_parameter_names: list[str] | str | None = None
    ) -> str:
        cov_matrix, par_names = self.get_cov_matrix(user_parameter_names)
        out = prettytable.PrettyTable()
        out.field_names = [""] + par_names.labels_for_all_scalar_elements

        # Calculate the smallest standard error of variances
        factor = np.sqrt(0.5 / (self.sample_count() - 1))
        se = np.sqrt(min(np.diag(cov_matrix))) * factor
        digits = int(np.ceil(-np.log10(se))) + 1

        cov_matrix_txt = np.round(cov_matrix, digits)

        for i in range(par_names.scalar_count):
            # Suppres scientific notation
            out.add_row(
                [par_names.labels_for_all_scalar_elements[i]]
                + [f"{cov_matrix_txt[i, j]:.4f}" for j in range(par_names.scalar_count)]
            )
        return str(out)

    def repr_with_sampling_errors(self):
        # Table example:
        #         mean   se_mean       sd       10%      90%
        # mu  7.751103 0.1113406 5.199004 1.3286256 14.03575
        # tau 6.806410 0.1785522 6.044944 0.9572097 14.48271
        out = repr(self.inference_engine) + "\n"

        out += self.formatted_runtime() + "\n"

        table = prettytable.PrettyTable()
        table.field_names = ["Parameter", "index", "mu", "sigma", "10%", "90%"]
        for flatindex in self.parameter_names.get_flatindices_from_parnames(None):
            par_name = self.parameter_names.get_parameter_from_index(flatindex)
            par_value: ValueWithError = self.get_parameter_estimate_flat(flatindex)
            par_label = self.parameter_names.pretty_shapeindex(
                self.parameter_names.flatindex_to_shapeindex(par_name, flatindex)
            )
            ci = par_value.get_CI(0.8)
            table.add_row(
                [
                    par_name,
                    par_label,
                    str(par_value.meanEstimate),
                    str(par_value.SEEstimate),
                    str(ci.pretty_lower),
                    str(ci.pretty_upper),
                ]
            )

        return out + str(table)

    def repr_without_sampling_errors(self):
        # Table example:
        #        value        10%      90%
        # mu  7.751103  1.3286256 14.03575
        # tau 6.806410  0.9572097 14.48271
        out = repr(self.inference_engine) + "\n"

        out += self.formatted_runtime() + "\n"

        table = prettytable.PrettyTable()
        table.field_names = ["Parameter", "index", "value", "10%", "90%"]
        for flatindex in self.parameter_names.get_flatindices_from_parnames(None):
            par_name = self.parameter_names.get_parameter_from_index(flatindex)
            par_value = self.get_parameter_estimate_flat(flatindex)
            par_label = self.parameter_names.pretty_shapeindex(
                self.parameter_names.flatindex_to_shapeindex(par_name, flatindex)
            )
            ci = par_value.get_CI(0.8)
            table.add_row(
                [
                    par_name,
                    par_label,
                    str(par_value.meanEstimate),
                    str(ci.pretty_lower),
                    str(ci.pretty_upper),
                ]
            )

        return out + str(table)

    def __repr__(self):
        if self.is_error():
            return super().__repr__()

        if self.model_name is not None:
            ans = f"{self.model_name}, "
        else:
            ans = ""
        ans += f"{self.inference_engine} running for {self.formatted_runtime():}\n"

        if self.sample_count is None:
            ans += self.repr_without_sampling_errors()
        else:
            ans += self.repr_with_sampling_errors()

        warnings = self.warnings
        if len(warnings) > 0:
            ans += "\nWarnings:\n"
            for warning in warnings:
                ans += warning + "\n"
        return ans
