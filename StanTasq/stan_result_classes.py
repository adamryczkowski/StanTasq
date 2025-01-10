import math

import numpy as np
import prettytable
from ValueWithError import ValueWithError
from overrides import overrides
from typing import Optional
from pydantic import BaseModel

from .ifaces import IInferenceResult, StanResultEngine


class StanResultMainEffects(IInferenceResult, BaseModel):
    one_dim_pars: dict[
        str, ValueWithError
    ]  # Parameter name in format "par" or "par[10][2][3]"
    par_dimensions: dict[
        str, list[int]
    ]  # base parameter name, without square brackets -> Shape of the parameter, needed to reconstruct the multi-dimensional parameters
    method_name: StanResultEngine
    calculation_sample_count: int

    def __init__(
        self,
        one_dim_pars: dict[str, ValueWithError],
        par_dimensions: dict[str, list[int]],
        method_name: StanResultEngine,
        calculation_sample_count: int,
    ):
        super().__init__(
            one_dim_pars=one_dim_pars,
            par_dimensions=par_dimensions,
            method_name=method_name,
            calculation_sample_count=calculation_sample_count,
        )

    @property
    @overrides()
    def one_dim_parameters_count(self) -> int:
        return len(self.one_dim_pars)

    @property
    @overrides
    def user_parameters(self) -> list[str]:
        return list(self.par_dimensions.keys())

    @property
    @overrides
    def onedim_parameters(self) -> list[str]:
        return list(self.one_dim_pars.keys())

    @overrides
    def sample_count(self, onedim_parameter_name: str = None) -> float | int | None:
        if onedim_parameter_name is None:
            return self.one_dim_pars[list(self.one_dim_pars.keys())[0]].N
        return self.sample_count

    @overrides
    def get_cov_matrix(
        self, user_parameter_names: list[str] | str | None = None
    ) -> Optional[tuple[np.ndarray, list[str]]]:
        return None

    @overrides
    def all_main_effects(self) -> dict[str, ValueWithError]:
        return self.one_dim_pars

    @overrides
    def draws(self, incl_raw: bool = True) -> np.ndarray | None:
        pass

    @overrides
    def __repr__(self):
        # Table example:
        #         mean   se_mean       sd       10%      90%
        # mu  7.751103 0.1113406 5.199004 1.3286256 14.03575
        # tau 6.806410 0.1785522 6.044944 0.9572097 14.48271

        table = prettytable.PrettyTable()
        table.field_names = ["Parameter", "index", "mu", "sigma", "10%", "90%"]
        for par, dims in self.par_dimensions.items():
            if len(dims) == 1 and dims[0] == 1:
                par_name = par
                par_value = self.one_dim_pars[par_name]
                ci = par_value.get_CI(0.8)
                table.add_row(
                    [
                        par_name,
                        "",
                        str(par_value.estimateMean),
                        str(par_value.estimateSE),
                        str(ci.pretty_lower),
                        str(ci.pretty_upper),
                    ]
                )
            else:
                max_idx = math.prod(dims)
                idx = [0 for _ in dims]
                i = 0
                while i < max_idx:
                    idx_txt = "[" + "][".join([str(i + 1) for i in idx]) + "]"
                    par_name = f"{par}{idx_txt}"
                    par_value = self.one_dim_pars[par_name]
                    ci = par_value.get_CI(0.8)
                    table.add_row(
                        [
                            par,
                            idx_txt,
                            str(par_value.estimateMean),
                            str(par_value.estimateSE),
                            str(ci.pretty_lower),
                            str(ci.pretty_upper),
                        ]
                    )
                    par = ""
                    i += 1
                    idx[-1] += 1
                    for j in range(len(dims) - 1, 0, -1):
                        if idx[j] >= dims[j]:
                            idx[j] = 0
                            idx[j - 1] += 1

        return str(table)

    @overrides
    def user_parameter_count(self) -> int:
        return len(self.par_dimensions)

    @overrides
    def get_parameter_shape(self, par_name: str) -> list[int]:
        return self.par_dimensions[par_name]

    @overrides
    def get_parameter_estimate(self, par_name: str, idx: list[int]) -> ValueWithError:
        if par_name not in self.par_dimensions:
            raise ValueError(f"Parameter {par_name} not found in the result")

        if len(self.par_dimensions[par_name]) == 1:
            return self.one_dim_pars[par_name]
        else:
            if isinstance(idx, int):
                idx = [idx]
            if len(idx) != len(self.par_dimensions[par_name]):
                raise ValueError(
                    f"Index {idx} has wrong length for parameter {par_name} with shape {self.par_dimensions[par_name]}"
                )

            one_par_name = par_name + "[" + "][".join([str(i) for i in idx]) + "]"
            return self.one_dim_pars[one_par_name]

    @overrides
    def get_parameter_mu(self, par_name: str) -> np.ndarray:
        if par_name not in self.par_dimensions:
            raise ValueError(f"Parameter {par_name} not found in the result")

        if len(self.par_dimensions[par_name]) == 1:
            return np.array([self.one_dim_pars[par_name].value])
        else:
            return np.array(
                [
                    self.one_dim_pars[
                        par_name + "[" + "][".join([str(i) for i in idx]) + "]"
                    ].value
                    for idx in np.ndindex(*self.par_dimensions[par_name])
                ]
            )

    @overrides
    def get_parameter_sigma(self, user_par_name: str) -> np.ndarray:
        if len(self.par_dimensions[user_par_name]) == 1:
            return np.array([self.one_dim_pars[user_par_name].SE])
        else:
            return np.array(
                [
                    self.one_dim_pars[
                        user_par_name + "[" + "][".join([str(i) for i in idx]) + "]"
                    ].SE
                    for idx in np.ndindex(*self.par_dimensions[user_par_name])
                ]
            )

    def mus(self) -> tuple[np.ndarray, dict[str, int]]:
        """Returns the means of the parameters"""
        par_count = len(self.one_dim_pars)
        ans = np.ndarray(par_count)
        keys = {}
        for i, (one_par, value) in enumerate(self.one_dim_pars.items()):
            ans[i] = value.value
            keys[one_par] = i
        return ans, keys


#
# class StanResultFull(StanResultMainEffects, IInferenceResult):
#     one_dim_pars: dict[str, ValueWithError]
#
#     def __init__(self, one_dim_pars: dict[str, ValueWithError], par_dimensions: dict[str, list[int]]):
#         # noinspection PyTypeChecker
#         super().__init__(one_dim_pars, par_dimensions)
#
#     @overrides
#     def get_covariances(self) -> tuple[np.ndarray, dict[str, int]]:
#         """Returns the covariance matrix of the parameters and a dictionary of the parameter names"""
#         par_names = list(self.par_dimensions.keys())
#         par_count = len(self._one_dim_pars)
#         cov = np.ndarray((par_count, par_count))
#         keys = {}
#         for i in range(par_count):
#             for j in range(i, par_count):
#                 cov[i, j] = np.cov(self._one_dim_pars[par_names[i]].vector, self._one_dim_pars[par_names[j]].vector)
#                 cov[j, i] = cov[i, j]
#                 keys[par_names[i]] = i
#         return cov, keys
#
#
# class StanResultMultiNormal(IInferenceResult):
#     _main_effects = np.ndarray
#     _covariances = np.ndarray
#     _keys = dict[str, int]
#     _par_dimensions: dict[str, list[int]]
#
#     def __init__(self, main_effects: np.ndarray, covariances: np.ndarray, keys: dict[str, int],
#                  par_dimensions: dict[str, list[int]]):
#         assert isinstance(main_effects, np.ndarray)
#         assert isinstance(covariances, np.ndarray)
#         assert isinstance(keys, dict)
#         assert isinstance(par_dimensions, dict)
#         assert main_effects.shape[0] == covariances.shape[0]
#         assert main_effects.shape[0] == covariances.shape[1]
#         assert main_effects.shape[0] == len(keys)
#
#         self._main_effects = main_effects
#         self._covariances = covariances
#         self._keys = keys
#         self._par_dimensions = par_dimensions
#
#     @overrides
#     def __repr__(self):
#         pass
#
#     @overrides
#     def user_parameter_count(self) -> int:
#         return len(self._keys)
#
#     @overrides
#     def get_parameter_shape(self, user_par_name: str) -> list[int]:
#         return self._par_dimensions[user_par_name]
#
#     @overrides
#     def get_parameter_estimate(self, onedim_par_name: str) -> ValueWithError:
#         if onedim_par_name not in self._keys:
#             raise ValueError(f"Parameter {onedim_par_name} not found in the result")
#         if idx != 0:
#             raise ValueError(f"Parameter {onedim_par_name} is not one-dimensional")
#         idx = self._keys[onedim_par_name]
#         return make_ValueWithError(mean=self._main_effects[idx], se=np.sqrt(self._covariances[idx, idx]))
#
#     @overrides
#     def get_parameter_mu(self, user_par_name: str) -> np.ndarray:
#         if self.get_parameter_shape(user_par_name) == [1]:
#             return np.array([self._main_effects[self._keys[user_par_name]]])
#
#         return np.array([self._main_effects[user_par_name + "[" + "][".join([str(i) for i in idx]) + "]"]
#                          for idx in np.ndindex(*self._par_dimensions[user_par_name])])
#
#     @overrides
#     def get_parameter_sigma(self, user_par_name: str) -> np.ndarray:
#         if self.get_parameter_shape(user_par_name) == [1]:
#             return np.array([np.sqrt(self._covariances[self._keys[user_par_name], self._keys[user_par_name]])])
#
#         return np.array([np.sqrt(self._covariances[user_par_name + "[" + "][".join([str(i) for i in idx]) + "]",
#                                                    user_par_name + "[" + "][".join([str(i) for i in idx]) + "]"])
#                          for idx in np.ndindex(*self._par_dimensions[user_par_name])])
#
#     @overrides
#     def get_covariances(self) -> tuple[np.ndarray, dict[str, int]]:
#         return self._covariances, self._keys
