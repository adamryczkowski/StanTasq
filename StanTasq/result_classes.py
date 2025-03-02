import datetime as dt
from typing import Literal
from typing import Optional

import numpy as np
from ValueWithError import (
    ValueWithError,
    make_ValueWithError,
    make_ValueWithError_from_vector,
    VectorOfValues,
)
from numpydantic import NDArray, Shape
from pydantic import model_validator
from .ifaces import StanErrorType, StanResultEngine
from .iresult import (
    ResultErrorBase,
    ResultBase,
    ParameterNames,
    Covariances,
    ResultClassType,
    I_ResultCovariances,
    I_ResultMainEffects,
    I_ResultSamples,
    ColumnsSelectorType,
)
from overrides import overrides


class ResultError(ResultErrorBase):
    """ """

    class_type: ResultClassType = Literal[ResultClassType.Error]

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
        super().__init__(
            error_type=error_type,
            inference_engine=inference_engine,
            inference_runtime=inference_runtime,
            compilation_time=compilation_time,
            worker_tag=worker_tag,
            error_message=error_message,
            model_name=model_name,
            warnings=warnings,
            **kwargs,
        )


class ResultMainEffects(ResultBase, I_ResultMainEffects):
    """ """

    class_type: ResultClassType = Literal[ResultClassType.MainEffects]
    main_effects_by_flatindex: list[ValueWithError]

    def __init__(
        self,
        parameter_names: ParameterNames,
        error_type: StanErrorType,
        inference_engine: StanResultEngine,
        inference_runtime: dt.timedelta,
        compilation_time: dt.timedelta,
        worker_tag: str,
        main_effects_by_flatindex: list[ValueWithError],
        error_message: Optional[str] = None,
        model_name: Optional[str] = None,
        warnings: list[str] = None,
        **kwargs,
    ):
        super().__init__(
            parameter_names=parameter_names,
            error_type=error_type,
            inference_engine=inference_engine,
            inference_runtime=inference_runtime,
            compilation_time=compilation_time,
            worker_tag=worker_tag,
            error_message=error_message,
            model_name=model_name,
            warnings=warnings,
            main_effects_by_flatindex=main_effects_by_flatindex,
            **kwargs,
        )

    @overrides
    def sample_count(self, flatindex: int) -> float | int:
        return self.main_effects_by_flatindex[flatindex].N

    @overrides
    def get_parameter_estimate_flat(self, flatindex: int) -> ValueWithError:
        return self.main_effects_by_flatindex[flatindex]

    @overrides
    def get_parameter_mu(
        self, par_names: list[str] | str | None = None
    ) -> NDArray[Shape["*, ..."], float]:  # noqa F722
        flat_indices = self.parameter_names.get_flatindices_from_parnames(par_names)
        ans = [self.get_parameter_estimate_flat(i).value for i in flat_indices]
        # noinspection PyTypeChecker
        return np.asarray(ans)

    @overrides
    def get_parameter_sigma(
        self, par_names: list[str] | str | None = None
    ) -> NDArray[Shape["*, ..."], float]:  # noqa F722
        flat_indices = self.parameter_names.get_flatindices_from_parnames(par_names)
        ans = [self.get_parameter_estimate_flat(i).SD for i in flat_indices]
        # noinspection PyTypeChecker
        return np.asarray(ans)


class ResultCovMatrix(ResultBase, I_ResultCovariances):
    class_type: ResultClassType = Literal[ResultClassType.Covariances]
    covariances: Covariances
    main_effects: list[float]
    Ns: list[float]

    def __init__(
        self,
        parameter_names: ParameterNames,
        error_type: StanErrorType,
        inference_engine: StanResultEngine,
        inference_runtime: dt.timedelta,
        compilation_time: dt.timedelta,
        worker_tag: str,
        covariances: Covariances,
        main_effects: list[float],
        Ns: list[float],
        error_message: Optional[str] = None,
        model_name: Optional[str] = None,
        warnings: list[str] = None,
        **kwargs,
    ):
        super().__init__(
            parameter_names=parameter_names,
            error_type=error_type,
            inference_engine=inference_engine,
            inference_runtime=inference_runtime,
            compilation_time=compilation_time,
            worker_tag=worker_tag,
            error_message=error_message,
            model_name=model_name,
            warnings=warnings,
            covariances=covariances,
            main_effects=main_effects,
            Ns=Ns,
            **kwargs,
        )

    @overrides
    def sample_count(self, flatindex: int) -> float | int:
        return self.Ns[flatindex]

    @overrides
    def get_parameter_estimate_flat(self, flatindex: int) -> ValueWithError:
        return make_ValueWithError(
            mean=self.main_effects[flatindex],
            SD=np.sqrt(self.covariances.matrix[flatindex, flatindex]),
            N=self.Ns[flatindex],
        )

    @overrides
    def get_parameter_mu(
        self, par_names: list[str] | str | None = None
    ) -> NDArray[Shape["*, ..."], float]:  # noqa F722
        par_indices = list(
            self.parameter_names.get_flatindices_from_parnames(par_names)
        )
        assert len(par_indices) > 0
        # TODO: possible error
        return self.main_effects[par_indices]

    def get_parameter_sigma(
        self, par_names: list[str] | str | None = None
    ) -> NDArray[Shape["*, ..."], float]:  # noqa F722
        par_indices = list(
            self.parameter_names.get_flatindices_from_parnames(par_names)
        )
        # TODO: possible error
        return self.covariances.matrix[par_indices, par_indices]

    @overrides
    @property
    def covariance_matrix(self) -> tuple[Covariances, ParameterNames]:
        return self.covariances, self.parameter_names


class ResultSamples(ResultBase, I_ResultSamples):
    samples: NDArray[Shape["*, ..."], float]  # noqa F722
    Ns: list[float | int]

    @model_validator(mode="after")
    def samples_column_count(self):
        assert self.samples.shape[0] > 0
        assert self.samples.shape[1] >= self.parameter_names.scalar_count
        # for flatname in self.parameter_names.flatnames:
        #     assert flatname in self.parameter_names.flatnames

    def __init__(
        self,
        parameter_names: ParameterNames,
        error_type: StanErrorType,
        inference_engine: StanResultEngine,
        inference_runtime: dt.timedelta,
        compilation_time: dt.timedelta,
        worker_tag: str,
        samples: NDArray[Shape["*, ..."], float],  # noqa F722
        Ns: list[float | int],
        error_message: Optional[str] = None,
        model_name: Optional[str] = None,
        warnings: list[str] = None,
        **kwargs,
    ):
        super().__init__(
            parameter_names=parameter_names,
            error_type=error_type,
            inference_engine=inference_engine,
            inference_runtime=inference_runtime,
            compilation_time=compilation_time,
            worker_tag=worker_tag,
            samples=samples,
            Ns=Ns,
            error_message=error_message,
            model_name=model_name,
            warnings=warnings,
            **kwargs,
        )

    @overrides
    def sample_count(self, flatindex: int) -> float | int:
        return self.Ns[flatindex]

    @overrides
    def get_raw_samples(self, par_names: ColumnsSelectorType = None) -> np.ndarray:
        return self.samples[
            :, self.parameter_names.get_flatindices_from_parnames(par_names)
        ]

    @overrides
    def get_parameter_estimate_flat(self, flatindex: int) -> ValueWithError:
        estimate_vector = self.get_estimate_with_samples(flatindex)
        return estimate_vector.get_ValueWithError(CI_levels=[0.99, 0.95, 0.9, 0.8, 0.5])

    @overrides
    def get_parameter_mu(
        self, par_names: list[str] | str | None = None
    ) -> NDArray[Shape["*, ..."], float]:  # noqa F722
        par_indices = self.parameter_names.get_flatindices_from_parnames(par_names)
        return np.mean(self.samples[:, par_indices], axis=0)

    @overrides
    def get_parameter_sigma(
        self, par_names: list[str] | str | None = None
    ) -> NDArray[Shape["*, ..."], float]:  # noqa F722
        par_indices = self.parameter_names.get_flatindices_from_parnames(par_names)
        return np.std(self.samples[:, par_indices], axis=0)

    @overrides
    @property
    def covariance_matrix(self) -> tuple[Covariances, ParameterNames]:
        # noinspection PyTypeChecker
        covariances = Covariances(
            matrix=np.cov(
                self.samples[
                    :, self.parameter_names.get_flatindices_from_parnames(None)
                ],
                rowvar=False,
            )
        )
        return covariances, self.parameter_names

    @overrides
    def get_estimate_with_samples(self, flatindex: int) -> VectorOfValues:
        raw_samples = self.get_raw_samples(flatindex)
        return make_ValueWithError_from_vector(vector=raw_samples, N=self.Ns[flatindex])
