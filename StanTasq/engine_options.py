from __future__ import annotations
from abc import ABC
from typing import Optional, Any, Mapping, Union, List
from entityhash import EntityHash, calc_hash
from .ifaces import StanResultEngine, StanOutputScope

from pydantic import BaseModel, model_validator
from cmdstanpy.model import SamplerArgs


class AbstractEngineOpts(ABC, BaseModel):
    engine: StanResultEngine


class MCMCEngineOpts(AbstractEngineOpts, BaseModel):
    sampler_args: SamplerArgsWrapper
    sig_figs: int = 6
    seed: Optional[int] = None
    local_details_scope: StanOutputScope = StanOutputScope.RawOutput  # What to put in the locally-serialized output file. User will not be able to get more detailed result than this without re-running the task.


class SamplerArgsWrapper(BaseModel):
    iter_warmup: Optional[int] = None
    iter_sampling: Optional[int] = None
    save_warmup: bool = False
    thin: Optional[int] = None
    max_treedepth: Optional[int] = None
    metric: Optional[str | dict[str, Any] | list[str] | list[dict[str, Any]]] = None
    step_size: Optional[float | list[float]] = None
    adapt_engaged: bool = True
    adapt_delta: Optional[float] = None
    adapt_init_phase: Optional[int] = None
    adapt_metric_window: Optional[int] = None
    adapt_step_size: Optional[int] = None
    fixed_param: bool = False
    inits: Union[
        Mapping[str, Any],
        float,
        str,
        List[str],
        List[Mapping[str, Any]],
        None,
    ] = None
    num_chains: int = 1

    def __init__(
        self,
        iter_warmup: Optional[int] = None,
        iter_sampling: Optional[int] = None,
        save_warmup: bool = False,
        thin: Optional[int] = None,
        max_treedepth: Optional[int] = None,
        metric: Optional[
            str | dict[str, Any] | dict[str] | list[dict[str, Any]]
        ] = None,
        step_size: Optional[float | list[float]] = None,
        adapt_engaged: bool = True,
        adapt_delta: Optional[float] = None,
        adapt_init_phase: Optional[int] = None,
        adapt_metric_window: Optional[int] = None,
        adapt_step_size: Optional[int] = None,
        fixed_param: bool = False,
        num_chains: int = 1,
        *args,
        **kwargs,
    ):
        super().__init__(
            iter_warmup=iter_warmup,
            iter_sampling=iter_sampling,
            save_warmup=save_warmup,
            thin=thin,
            max_treedepth=max_treedepth,
            metric=metric,
            step_size=step_size,
            adapt_engaged=adapt_engaged,
            adapt_delta=adapt_delta,
            adapt_init_phase=adapt_init_phase,
            adapt_metric_window=adapt_metric_window,
            adapt_step_size=adapt_step_size,
            fixed_param=fixed_param,
            num_chains=num_chains,
            *args,
            **kwargs,
        )

    @model_validator(mode="after")
    def validate_model(self):
        self.SamplerArgs.validate(self.num_chains)

    @property
    def SamplerArgs(self) -> SamplerArgs:
        return SamplerArgs(**self.model_dump(mode="python"))

    @property
    def sampling_options_hash(self) -> EntityHash:
        return calc_hash(self.model_dump(mode="python"))
