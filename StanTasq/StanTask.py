from __future__ import annotations
from .ifaces import StanResultEngine, StanOutputScope
from pydantic import BaseModel, Field
from .normalized_model import NormalizedModel
from .data import DataSet
from .engine_options import MCMCEngineOpts
from entityhash import EntityHash, calc_hash
from typing import Union


class StanTask(BaseModel):
    """
    Class that represents all data pertaining to a Stan task.
    This class is meant as a data container for the StanTaskBroker, not a user-facing class that performs any computation.
    """

    model: NormalizedModel
    data: DataSet
    cpp_opts: dict[str, str] = {}
    engine: StanResultEngine
    output_scope: StanOutputScope
    compress_values_with_errors: bool  # Replaces samples with standard errors and confidence intervals at 10%. In future, we may add a 3rd option, a histogram of the samples.
    sampling_options: Union[MCMCEngineOpts] = Field(discriminator="engine")

    @property
    def task_hash(self) -> EntityHash:
        return calc_hash(self.model_dump(mode="python"))


#
#
# def serial_compute_model(
#     model_code: str,
#     data: dict[str, Any],
#     model_name: str,
#     engine: StanResultEngine,
#     output_scope: StanOutputScope,
#     compress_values_with_errors: bool,
#     worker_tag: str,
#     **kwargs: Any,
# ) -> IInferenceResult:
#     model_cache_dir = Path(__file__).parent / "model_cache"
#
#     runner = CmdStanRunner(model_cache=model_cache_dir, worker_tag=worker_tag)
#     runner.install_dependencies()
#
#     time_start = dt.datetime.now()
#     runner.load_model_by_str(model_code, model_name)
#     if not runner.is_model_loaded:
#         print(runner.model_code)
#         print(runner.messages["stanc_error"])
#         return StanErroneousResult(
#             method_name=engine,
#             runtime=(dt.datetime.now() - time_start),
#             messages=runner.messages,
#             worker_tag=worker_tag,
#             error_type=StanErrorType.SYNTAX_ERROR,
#         )
#     runner.compile_model()
#     if not runner.is_model_compiled:
#         print(runner.messages["compile_error"])
#         return StanErroneousResult(
#             method_name=engine,
#             runtime=(dt.datetime.now() - time_start),
#             messages=runner.messages,
#             worker_tag=worker_tag,
#             error_type=StanErrorType.COMPILE_ERROR,
#         )
#     runner.load_data_by_dict(data)
#     if not runner.is_data_set:
#         print(runner.messages["data_error"])
#         return StanErroneousResult(
#             method_name=engine,
#             runtime=(dt.datetime.now() - time_start),
#             messages=runner.messages,
#             worker_tag=worker_tag,
#             error_type=StanErrorType.DATA_ERROR,
#         )
#     if engine == StanResultEngine.MCMC:
#         result = runner.sampling(iter_sampling=1000, num_chains=8, **kwargs)
#     elif engine == StanResultEngine.LAPLACE:
#         result = runner.laplace_sample(output_samples=1000, **kwargs)
#     elif engine == StanResultEngine.VB:
#         result = runner.variational_bayes(output_samples=1000, **kwargs)
#     elif engine == StanResultEngine.PATHFINDER:
#         result = runner.pathfinder(output_samples=1000, **kwargs)
#     else:
#         runner.messages["error"] = f"Unknown runner engine: {engine}"
#         return StanErroneousResult(
#             method_name=engine,
#             runtime=(dt.datetime.now() - time_start),
#             messages=runner.messages,
#             worker_tag=worker_tag,
#         )
#     assert isinstance(result, InferenceResult)
#     # print(messages)
#     out = result.get_serializable_version(
#         output_scope=output_scope,
#         compress_values_with_errors=compress_values_with_errors,
#     )
#     return out
#
#
# @broker.task
# async def compute_model(
#     model_code: str,
#     data: dict[str, Any],
#     model_name: str,
#     engine: StanResultEngine,
#     output_scope: StanOutputScope,
#     compress_values_with_errors: bool,
#     context: Annotated[Context, TaskiqDepends()],
# ) -> Optional[str]:
#     if context is None:
#         import socket
#
#         worker_tag = socket.gethostname()
#     else:
#         worker_tag = context.state.worker_tag
#
#     return serial_compute_model(
#         model_code=model_code,
#         data=data,
#         model_name=model_name,
#         engine=engine,
#         output_scope=output_scope,
#         compress_values_with_errors=compress_values_with_errors,
#         worker_tag=worker_tag,
#     )
