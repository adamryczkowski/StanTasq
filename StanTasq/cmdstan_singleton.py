import io
import subprocess
from _datetime import datetime
from contextlib import redirect_stdout, redirect_stderr
from multiprocessing import cpu_count
from pathlib import Path
from subprocess import run
from typing import Optional
from entityhash import EntityHash
from cachemanager import ObjectCache

import cmdstanpy

# import jsonpickle
from overrides import overrides

from .ifaces import ILocalInferenceResult
from .result_adapter import InferenceResult

# _fallback = json._default_encoder.default
# json._default_encoder.default = lambda obj: getattr(obj.__class__, "to_json", _fallback)(obj)

stan_CI_levels_dict = {0: 0.95, 1: 0.9, 2: 0.8, 3: 0.5}


def initialize(cpu_cores: int = 1):
    # Check if `stanc` is installed in the correct path:
    stanc_expected_path = Path(cmdstanpy.cmdstan_path()) / "bin" / "stanc"
    if stanc_expected_path.exists():
        # Run stanc and get its version
        output = run([str(stanc_expected_path), "--version"], capture_output=True)
        if output.returncode == 0:
            # Check if stdoutput starts with "stanc3"
            if output.stdout.decode().strip().startswith("stanc3"):
                return

    cmdstanpy.install_cmdstan(verbose=True, overwrite=False, cores=cpu_cores)


class CmdStanSingleton:
    _number_of_cores: int
    _output_dir: Path
    _initialized: bool

    _last_model_hash: EntityHash
    _worker_tag: str
    _number_of_cores: int

    _allow_optimizations_for_stanc: bool
    _stan_threads: bool

    _model_code_cache_manager: ObjectCache
    _model_data_cache_manager: ObjectCache
    _model_result_cache_manager: ObjectCache

    def __init__(
        self,
        model_cache: Path,
        data_cache: Path,
        result_cache: Path,
        worker_tag: str,
        number_of_cores: int = None,
        allow_optimizations_for_stanc: bool = True,
        stan_threads: bool = True,
    ):
        self._worker_tag = worker_tag
        self._model_code_cache_manager = ObjectCache.InitCache(model_cache)
        self._model_data_cache_manager = ObjectCache.InitCache(data_cache)
        self._model_result_cache_manager = ObjectCache.InitCache(result_cache)

        if number_of_cores is not None:
            assert isinstance(number_of_cores, int)
            assert number_of_cores > 0

        self._number_of_cores = number_of_cores

        self._allow_optimizations_for_stanc = allow_optimizations_for_stanc
        self._stan_threads = stan_threads

        self._initialized = False

    def install_dependencies(self):
        if self._initialized:
            return

        initialize(self.number_of_cores)
        self._initialized = True

    @property
    def number_of_cores(self) -> int:
        return cpu_count() if self._number_of_cores is None else self._number_of_cores

    # @property
    # @overrides
    # def error_state(self) -> StanErrorType:
    #     if "stanc_error" in self._messages:
    #         return StanErrorType.SYNTAX_ERROR
    #     if "compile_error" in self._messages:
    #         return StanErrorType.COMPILE_ERROR
    #     return StanErrorType.NO_ERROR

    def is_initialized(self) -> bool:
        return self._initialized

    @property
    def model_cache(self) -> ObjectCache:
        return self._model_code_cache_manager

    @property
    def data_cache(self) -> ObjectCache:
        return self._model_data_cache_manager

    @property
    def result_cache(self) -> ObjectCache:
        return self._model_result_cache_manager

    @property
    def worker_tag(self) -> str:
        return self._worker_tag

    # @overrides
    # def load_data_by_file(self, data_file: str | Path):
    #     """Loads data from a structureal file. Right now, the supported format is only JSON."""
    #     if isinstance(data_file, str):
    #         data_file = Path(data_file)
    #
    #     assert isinstance(data_file, Path)
    #     assert data_file.exists()
    #     assert data_file.is_file()
    #
    #     if data_file.suffix == ".json":
    #         json_data = data_file.read_text()
    #         self._data = json.loads(json_data)
    #     elif data_file.suffix == ".pkl":
    #         import pickle
    #
    #         with data_file.open("rb") as f:
    #             self._data = pickle.load(f)
    #     else:
    #         raise ValueError("Unknown data file type")
    #
    # @overrides
    # def load_data_by_dict(self, data: dict[str, float | int | np.ndarray]):
    #     assert isinstance(data, dict)
    #     self._data = data

    # @overrides
    # def sampling(
    #     self,
    #     num_chains: int,
    #     iter_sampling: int = None,
    #     iter_warmup: int = None,
    #     thin: int = 1,
    #     max_treedepth: int = None,
    #     seed: int = None,
    #     inits: dict[str, Any] | float | list[str] = None,
    # ) -> ILocalInferenceResult:
    #     assert self.is_model_compiled
    #
    #     threads_per_chain = 1
    #     parallel_chains = min(num_chains, self.number_of_cores)
    #     if "STAN_THREADS" in self._cpp_opts and self._cpp_opts["STAN_THREADS"]:
    #         if self.number_of_cores > num_chains:
    #             threads_per_chain = self.number_of_cores // num_chains
    #
    #     stdout = io.StringIO()
    #     stderr = io.StringIO()
    #
    #     now1 = datetime.now()
    #
    #     with redirect_stdout(stdout), redirect_stderr(stderr):
    #         try:
    #             ans = self._stan_model.sample(
    #                 data=self._data,
    #                 chains=num_chains,
    #                 parallel_chains=parallel_chains,
    #                 threads_per_chain=threads_per_chain,
    #                 seed=seed,
    #                 inits=inits,
    #                 iter_warmup=iter_warmup,
    #                 iter_sampling=iter_sampling,
    #                 thin=thin,
    #                 max_treedepth=max_treedepth,
    #                 output_dir=self._output_dir.name,
    #                 sig_figs=self._other_opts.get("sig_figs", None),
    #             )
    #         except subprocess.CalledProcessError as e:
    #             now2 = datetime.now()
    #             messages = {
    #                 "stdout": stdout.getvalue(),
    #                 "stderr": stderr.getvalue(),
    #                 "error": str(e),
    #             }
    #             return StanErroneousResult(
    #                 method_name=StanResultEngine.MCMC,
    #                 runtime=(dt.datetime.now() - now1),
    #                 messages=messages,
    #                 worker_tag=self._worker_tag,
    #                 error_type=StanErrorType.RUNTIME_ERROR,
    #             )
    #
    #     now2 = datetime.now()
    #
    #     messages = {"stdout": stdout.getvalue(), "stderr": stderr.getvalue()}
    #     obj = InferenceResult(
    #         ans, messages, runtime=now2 - now1, worker_tag=self._worker_tag
    #     )
    #     # out = obj.get_serializable_version(StanOutputScope.MainEffects)
    #     return obj

    @overrides
    def variational_bayes(
        self, output_samples: int = 1000, **kwargs
    ) -> ILocalInferenceResult:
        assert self.is_model_compiled

        stdout = io.StringIO()
        stderr = io.StringIO()

        now1 = datetime.now()

        with redirect_stdout(stdout), redirect_stderr(stderr):
            try:
                ans = self._stan_model.variational(
                    data=self._data,
                    output_dir=self._output_dir.name,
                    sig_figs=self._other_opts.get("sig_figs", None),
                    draws=output_samples,
                    **kwargs,
                )
            except Exception as e:
                now2 = datetime.now()
                messages = {
                    "stdout": stdout.getvalue(),
                    "stderr": stderr.getvalue(),
                    "error": str(e),
                }
                return InferenceResult(
                    None, messages, worker_tag=self._worker_tag, runtime=now2 - now1
                )
        now2 = datetime.now()
        messages = {"stdout": stdout.getvalue(), "stderr": stderr.getvalue()}
        out = InferenceResult(
            ans, messages, runtime=now2 - now1, worker_tag=self._worker_tag
        )

        return out

    @overrides
    def pathfinder(self, output_samples: int = 1000, **kwargs) -> ILocalInferenceResult:
        assert self.is_model_compiled

        stdout = io.StringIO()
        stderr = io.StringIO()

        now1 = datetime.now()

        with redirect_stdout(stdout), redirect_stderr(stderr):
            try:
                ans = self._stan_model.pathfinder(
                    data=self._data,
                    draws=output_samples,
                    output_dir=self._output_dir.name,
                    sig_figs=self._other_opts.get("sig_figs", None),
                    **kwargs,
                )
            except subprocess.CalledProcessError as e:
                now2 = datetime.now()
                messages = {
                    "stdout": stdout.getvalue(),
                    "stderr": stderr.getvalue(),
                    "error": str(e),
                }
                return InferenceResult(None, messages, worker_tag=self._worker_tag)

        now2 = datetime.now()

        messages = {"stdout": stdout.getvalue(), "stderr": stderr.getvalue()}
        out = InferenceResult(
            ans, messages, runtime=now2 - now1, worker_tag=self._worker_tag
        )
        return out

    @overrides
    def laplace_sample(
        self, output_samples: int = 1000, **kwargs
    ) -> ILocalInferenceResult:
        assert self.is_model_compiled

        stdout = io.StringIO()
        stderr = io.StringIO()

        now1 = datetime.now()

        with redirect_stdout(stdout), redirect_stderr(stderr):
            try:
                ans = self._stan_model.laplace_sample(
                    data=self._data,
                    output_dir=self._output_dir.name,
                    sig_figs=self._other_opts.get("sig_figs", None),
                    draws=output_samples,
                    **kwargs,
                )
            except subprocess.CalledProcessError as e:
                now2 = datetime.now()
                messages = {
                    "stdout": stdout.getvalue(),
                    "stderr": stderr.getvalue(),
                    "error": str(e),
                }
                return InferenceResult(
                    None, messages, worker_tag=self._worker_tag, runtime=now2 - now1
                )

        now2 = datetime.now()
        messages = {"stdout": stdout.getvalue(), "stderr": stderr.getvalue()}
        out = InferenceResult(
            ans, messages, runtime=now2 - now1, worker_tag=self._worker_tag
        )
        return out

    def optimize(
        self, **kwargs
    ) -> tuple[Optional[cmdstanpy.CmdStanMLE], dict[str, str]]:
        assert self.is_model_compiled

        stdout = io.StringIO()
        stderr = io.StringIO()

        now1 = datetime.now()

        with redirect_stdout(stdout), redirect_stderr(stderr):
            try:
                ans = self._stan_model.optimize(
                    data=self._data,
                    output_dir=self._output_dir.name,
                    sig_figs=self._other_opts.get("sig_figs", None),
                    **kwargs,
                )
            except subprocess.CalledProcessError as e:
                now2 = datetime.now()
                messages = {
                    "stdout": stdout.getvalue(),
                    "stderr": stderr.getvalue(),
                    "error": str(e),
                    "runtime": now2 - now1,
                }
                return None, messages

        now2 = datetime.now()
        return ans, {
            "stdout": stdout.getvalue(),
            "stderr": stderr.getvalue(),
            "runtime": now2 - now1,
        }

    @property
    def model_code(self) -> str | None:
        if self._model_filename is not None:
            with self._model_filename.open("r") as f:
                return f.read()
        return None

    # def to_json(self) -> str:
    #     return jsonpickle.encode(self)
