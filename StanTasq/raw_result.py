from __future__ import annotations

import datetime as dt

# import json
import pickle
from typing import Literal, Optional

# import jsonpickle
import numpy as np
from ValueWithError import (
    ValueWithError,
    make_ValueWithError_from_vector,
    VectorOfValues,
)
from cmdstanpy import CmdStanLaplace, CmdStanVB, CmdStanMCMC, CmdStanPathfinder
from cmdstanpy.cmdstan_args import CmdStanArgs
from cmdstanpy.stanfit.metadata import InferenceMetadata
from cmdstanpy.stanfit.vb import RunSet
from overrides import overrides
from pydantic import computed_field

from .ifaces import StanErrorType, StanResultEngine
from .iresult import (
    ResultBase,
    ResultClassType,
    ParameterNames,
    I_ResultSamples,
    ColumnsSelectorType,
    Covariances,
)


class ResultRaw(ResultBase, I_ResultSamples):
    class_type: ResultClassType = Literal[ResultClassType.Raw]
    _result: CmdStanLaplace | CmdStanVB | CmdStanMCMC | CmdStanPathfinder
    _cached_draws: Optional[np.ndarray] = None

    def __init__(
        self,
        serialized_result: bytes,
        inference_engine: StanResultEngine,
        # result: CmdStanLaplace | CmdStanVB | CmdStanMCMC | CmdStanPathfinder,
        messages: dict[str, str],
        worker_tag: str,
        inference_runtime: dt.timedelta,
        compilation_time: dt.timedelta,
        warnings: list[str],
        model_name: str,
        **kwargs,
    ) -> None:
        assert serialized_result is not None
        result = ResultRaw.Deserialize(serialized_result)
        if isinstance(result, CmdStanLaplace):
            inference_engine2 = StanResultEngine.LAPLACE
        elif isinstance(result, CmdStanVB):
            inference_engine2 = StanResultEngine.VB
        elif isinstance(result, CmdStanMCMC):
            inference_engine2 = StanResultEngine.MCMC
        elif isinstance(result, CmdStanPathfinder):
            inference_engine2 = StanResultEngine.PATHFINDER
        else:
            raise ValueError("Unknown result type")
        assert inference_engine2 == inference_engine
        self._result = result

        parameter_names = self._infer_parameters(result)

        super().__init__(
            parameter_names=parameter_names,
            error_type=StanErrorType.NO_ERROR,
            inference_engine=inference_engine,
            inference_runtime=inference_runtime,
            compilation_time=compilation_time,
            worker_tag=worker_tag,
            error_message=None,
            model_name=model_name,
            warnings=warnings,
            **kwargs,
        )
        self._cached_draws = None

    @staticmethod
    def _infer_parameters(
        raw_result: CmdStanLaplace | CmdStanVB | CmdStanMCMC | CmdStanPathfinder,
    ) -> ParameterNames:
        metadata: InferenceMetadata = raw_result.metadata
        # parameter_names = list(metadata.stan_vars.keys())
        parameter_shapes = {
            parameter_name: parameter_meta.dimensions
            for parameter_name, parameter_meta in metadata.stan_vars.items()
        }
        return ParameterNames(parameter_shapes=parameter_shapes)

    @overrides
    def sample_count(self, flatindex: int) -> float | int:
        if self.inference_engine == StanResultEngine.LAPLACE:
            return self.draws.shape[0]
        elif self.inference_engine == StanResultEngine.VB:
            return self.draws.shape[0]
        elif self.inference_engine == StanResultEngine.MCMC:
            s = self._result.summary()
            return s["ESS_bulk"][flatindex]
        elif self.inference_engine == StanResultEngine.PATHFINDER:
            return self.draws.shape[0]
        else:
            raise ValueError("Unknown result type")

    @overrides
    def get_raw_samples(self, par_names: ColumnsSelectorType = None) -> np.ndarray:
        flatindices = self.parameter_names.get_flatindices_from_parnames(par_names)
        return self.all_parameter_draws[..., flatindices]

    @property
    def draws(self) -> np.ndarray:
        if self._cached_draws is None:
            if self.result_type == StanResultEngine.LAPLACE:
                self._cached_draws = self._result.draws()
            elif self.result_type == StanResultEngine.VB:
                self._cached_draws = self._result.variational_sample
            elif self.result_type == StanResultEngine.MCMC:
                self._cached_draws = self._result.draws(concat_chains=True)
            elif self.result_type == StanResultEngine.PATHFINDER:
                self._cached_draws = self._result.draws()
            else:
                raise ValueError("Unknown result type")
        return self._cached_draws

    @overrides
    def get_special_raw_samples(self) -> Optional[np.ndarray]:
        if self.result_type == StanResultEngine.LAPLACE:
            return self.draws[:, :2]
        elif self.result_type == StanResultEngine.VB:
            return self.draws[:, :3]
        elif self.result_type == StanResultEngine.MCMC:
            return self.draws[:, :7]
        elif self.result_type == StanResultEngine.PATHFINDER:
            return self.draws[:, :2]
        else:
            raise ValueError("Unknown result type")

    @property
    def all_parameter_draws(self) -> Optional[np.ndarray]:
        if self.result_type == StanResultEngine.LAPLACE:
            return self.draws[:, 2:]
        elif self.result_type == StanResultEngine.VB:
            return self.draws[:, 3:]
        elif self.result_type == StanResultEngine.MCMC:
            return self.draws[:, 7:]
        elif self.result_type == StanResultEngine.PATHFINDER:
            return self.draws[:, 2:]
        else:
            raise ValueError("Unknown result type")

    @overrides
    def get_estimate_with_samples(self, flatindex: int) -> VectorOfValues:
        draws = self.get_parameter_draws()
        return make_ValueWithError_from_vector(
            vector=draws[:, flatindex], N=self.sample_count(flatindex)
        )

    @overrides
    def get_parameter_estimate_flat(self, flatindex: int) -> ValueWithError:
        estimate_vector = self.get_estimate_with_samples(flatindex)
        return estimate_vector.get_ValueWithError(CI_levels=[0.99, 0.95, 0.9, 0.8, 0.5])

    # @overrides
    # def get_draws(self, incl_raw: bool = True) -> np.ndarray:
    #     if self._draws is None:
    #         if self.result_type == StanResultEngine.NONE:
    #             return np.array([])
    #         elif self.result_type == StanResultEngine.LAPLACE:
    #             self._draws = self._result.draws()
    #         elif self.result_type == StanResultEngine.VB:
    #             self._draws = self._result.variational_sample
    #         elif self.result_type == StanResultEngine.MCMC:
    #             self._draws = self._result.draws(concat_chains=True)
    #         elif self.result_type == StanResultEngine.PATHFINDER:
    #             self._draws = self._result.draws()
    #         else:
    #             raise ValueError("Unknown result type")
    #
    #     if not incl_raw:
    #         if self.result_type == StanResultEngine.NONE:
    #             return np.array([])
    #         if self.result_type == StanResultEngine.LAPLACE:
    #             return self._draws[:, 2:]
    #         elif self.result_type == StanResultEngine.VB:
    #             return self._draws[:, 3:]
    #         elif self.result_type == StanResultEngine.MCMC:
    #             return self._draws[:, 7:]
    #         elif self.result_type == StanResultEngine.PATHFINDER:
    #             return self._draws[:, 2:]
    #         else:
    #             raise ValueError("Unknown result type")
    #
    #     return self._draws

    @overrides
    def get_parameter_sigma(self, par_names: ColumnsSelectorType = None) -> np.ndarray:
        samples = self.get_raw_samples(par_names)
        return np.std(samples, axis=0)

    @overrides
    @property
    def covariance_matrix(self) -> tuple[Covariances, ParameterNames]:
        data = self.get_raw_samples()
        indices = range(data.shape[1])
        # noinspection PyTypeChecker
        covariances = Covariances(matrix=np.cov(data[:, indices], rowvar=False))

        return covariances, self.parameter_names

    @computed_field
    def serialized_result(self) -> bytes:
        """Serializes the method into a single file that is readable."""
        try:
            rs: RunSet = self._result.runset
        except AttributeError:
            try:
                rs = self._result._runset
            except AttributeError:
                raise ValueError("No result available")

        a: CmdStanArgs = rs._args
        obj = {"runset": rs}
        if self.result_type == StanResultEngine.LAPLACE:
            obj["laplace_mode"] = self._result.mode

        pickled = pickle.dumps(a)

        return pickled

    @staticmethod
    def Deserialize(
        serialized_result: bytes,
    ) -> CmdStanLaplace | CmdStanVB | CmdStanMCMC | CmdStanPathfinder:
        # dest_dir = Path(tempfile.TemporaryDirectory().name)

        # Unzip the zip_path into the dest_dir

        obj: dict = pickle.loads(serialized_result)
        rs2: RunSet = obj["runset"]
        # rs2._args.output_dir = str(dest_dir)
        # rs2._csv_files = [str(dest_dir / Path(item).name) for item in rs2.csv_files]
        # rs2._stdout_files = [
        #     str(dest_dir / Path(item).name) for item in rs2.stdout_files
        # ]

        # 'SAMPLE', 'VARIATIONAL', 'LAPLACE', PATHFINDER
        if rs2._args.method.name == "SAMPLE":
            stanObj = CmdStanMCMC(rs2)
        elif rs2._args.method.name == "VARIATIONAL":
            stanObj = CmdStanVB(rs2)
        elif rs2._args.method.name == "LAPLACE":
            stanObj = CmdStanLaplace(rs2, mode=obj["laplace_mode"])
        elif rs2._args.method.name == "PATHFINDER":
            stanObj = CmdStanPathfinder(rs2)
        else:
            raise ValueError("Unknown method")

        return stanObj
