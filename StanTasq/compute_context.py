from .cmdstan_singleton import CmdStanSingleton
from .StanTask import StanTask
import cmdstanpy
from contextlib import redirect_stdout, redirect_stderr
from typing import Optional
from pathlib import Path
from .ifaces import StanResultEngine, StanErrorType, StanOutputScope
import io
import subprocess
import datetime as dt

from .result_adapter import InferenceResult
from .engine_options import MCMCEngineOpts
from .result_classes import ResultError, ResultMainEffects
from .iresult import ResultErrorBase


class StanComputeContext:
    """
    Class that collects all data relevant to the computation of a single Stan task.
    """

    stan_runner: CmdStanSingleton  # A singleton that holds the global state
    task: StanTask  # The task to be computed
    _compiled_model: Optional[cmdstanpy.CmdStanModel]
    _model_name: str
    _compilation_time: dt.timedelta
    _messages: dict[str, str] = {}

    def __init__(self, stan_state: CmdStanSingleton, task: StanTask, model_name: str):
        assert isinstance(stan_state, CmdStanSingleton)
        assert isinstance(task, StanTask)
        self.stan_runner = stan_state
        self.task = task
        self._compiled_model = None
        self._model = None
        self._model_name = model_name
        self._compiled_model = None

    def is_model_compiled(self) -> bool:
        return self._compiled_model is not None

    @property
    def compiled_model_exe_filename(self) -> Path:
        code_hash = self.task.model.model_hash
        return self.stan_runner.model_cache.get_item_filename(code_hash)

    @property
    def model_source_filename(self) -> Optional[Path]:
        return self.task.model.model_path

    def compile_model(self, force_recompile: bool = False):
        if self.is_model_compiled:
            return

        exe_cache_item = self.stan_runner.model_cache.get_object_by_hash(
            self.task.model.model_hash
        )
        if exe_cache_item is None:
            exe_file = self.stan_runner.model_cache.get_item_filename(
                self.task.model.model_hash
            )
            force_recompile = True
            exe_file.touch()
        else:
            exe_file = exe_cache_item.filename

        stdout = io.StringIO()
        stderr = io.StringIO()
        self.stan_runner.model_cache.prune_cache()
        time_start = dt.datetime.now()
        with redirect_stdout(stdout), redirect_stderr(stderr):
            try:
                model = cmdstanpy.CmdStanModel(
                    stan_file=self.task.model.model_path,
                    exe_file=str(exe_file),
                    force_compile=force_recompile,
                    stanc_options=self.task.model.stanc_opts,
                    cpp_options=self.task.cpp_opts,
                )
                model.compile(force=force_recompile)
            except subprocess.CalledProcessError as e:
                self._messages["compile_output"] = stdout.getvalue() + e.stdout
                self._messages["compile_warning"] = stderr.getvalue()
                self._messages["compile_error"] = e.stderr
                return
        self.stan_runner.model_cache.store_object(
            exe_file,
            self.task.model.model_hash,
            compute_time=(dt.datetime.now() - time_start).total_seconds() / 60,
            weight=1,
            object_size=exe_file.stat().st_size / (1024 * 1024 * 1024),
            force_store=True,
        )

        self._messages["compile_output"] = stdout.getvalue()
        self._messages["compile_warning"] = stderr.getvalue()
        self._compiled_model = model

    def sampling(
        self,
        compilation_time: dt.timedelta,
        # num_chains: int,
        # iter_sampling: int = None,
        # iter_warmup: int = None,
        # thin: int = 1,
        # max_treedepth: int = None,
        # seed: int = None,
        # inits: dict[str, Any] | float | list[str] = None,
    ) -> ResultErrorBase:
        assert self.is_model_compiled
        assert self.task.engine == StanResultEngine.MCMC
        options = self.task.sampling_options
        assert isinstance(options, MCMCEngineOpts)
        if self.stan_runner.result_cache.get_object_by_hash(self.task.task_hash):
            return self.stan_runner.result_cache.get_object_by_hash(self.task.task_hash)

        threads_per_chain = 1
        num_chains = options.sampler_args.num_chains
        number_of_cores = self.stan_runner.number_of_cores
        parallel_chains = min(num_chains, number_of_cores)
        seed = options.seed
        inits = options.sampler_args.inits
        iter_warmup = options.sampler_args.iter_warmup
        iter_sampling = options.sampler_args.iter_sampling
        thin = options.sampler_args.thin
        max_treedepth = options.sampler_args.max_treedepth
        if "STAN_THREADS" in self.task.cpp_opts and self.task.cpp_opts["STAN_THREADS"]:
            if number_of_cores > num_chains:
                threads_per_chain = number_of_cores // num_chains

        stdout = io.StringIO()
        stderr = io.StringIO()

        now1 = dt.datetime.now()

        with redirect_stdout(stdout), redirect_stderr(stderr):
            try:
                ans = self._compiled_model.sample(
                    data=self.task.data.data,
                    chains=num_chains,
                    parallel_chains=parallel_chains,
                    threads_per_chain=threads_per_chain,
                    seed=seed,
                    inits=inits,
                    iter_warmup=iter_warmup,
                    iter_sampling=iter_sampling,
                    thin=thin,
                    max_treedepth=max_treedepth,
                    output_dir=self.stan_runner.result_cache.get_item_filename(
                        self.task.task_hash
                    ),
                    sig_figs=options.sig_figs,
                )
            except subprocess.CalledProcessError as e:
                messages = {
                    "stdout": stdout.getvalue(),
                    "stderr": stderr.getvalue(),
                    "error": str(e),
                }
                return ResultError(
                    error_type=StanErrorType.RUNTIME_ERROR,
                    inference_engine=StanResultEngine.MCMC,
                    inference_runtime=dt.datetime.now() - now1,
                    compilation_time=compilation_time,
                    worker_tag=self.stan_runner.worker_tag,
                    error_message=messages,
                    model_name=self._model_name,
                    warnings=[],
                )

        messages = {"stdout": stdout.getvalue(), "stderr": stderr.getvalue()}
        max_scope = self.task.output_scope
        if max_scope == StanOutputScope.MainEffects:
            return ResultMainEffects(
                parameter_names=ans.column_names,
            )

        obj = InferenceResult(
            ans, messages, runtime=dt.datetime.now() - now1, worker_tag=self._worker_tag
        )
        # out = obj.get_serializable_version(StanOutputScope.MainEffects)
        return obj
