from .broker import broker
from taskiq import Context, TaskiqDepends
from typing import Annotated, Any, Optional
from pathlib import Path
from .cmdstan_runner import CmdStanRunner, InferenceResult
from .ifaces import StanResultEngine, StanOutputScope
import datetime as dt


def serial_compute_model(
    model_code: str,
    data: dict[str, Any],
    model_name: str,
    engine: StanResultEngine,
    output_scope: StanOutputScope,
    compress_values_with_errors: bool,
    worker_tag: str,
    **kwargs: Any,
) -> Optional[str]:
    model_cache_dir = Path(__file__).parent / "model_cache"

    runner = CmdStanRunner(model_cache=model_cache_dir, worker_tag=worker_tag)
    runner.install_dependencies()

    time_start = dt.datetime.now()
    runner.load_model_by_str(model_code, model_name)
    if not runner.is_model_loaded:
        print(runner.model_code)
        print(runner.messages["stanc_error"])

        return InferenceResult(
            None, runner.messages, runtime=(dt.datetime.now() - time_start)
        )
    runner.compile_model()
    if not runner.is_model_compiled:
        print(runner.messages["compile_error"])
        return InferenceResult(
            None, runner.messages, runtime=(dt.datetime.now() - time_start)
        )
    runner.load_data_by_dict(data)
    if not runner.is_data_set:
        print(runner.messages["data_error"])
        return InferenceResult(
            None, runner.messages, runtime=(dt.datetime.now() - time_start)
        )
    if engine == StanResultEngine.MCMC:
        result = runner.sampling(iter_sampling=1000, num_chains=8, **kwargs)
    elif engine == StanResultEngine.LAPLACE:
        result = runner.laplace_sample(output_samples=1000, **kwargs)
    elif engine == StanResultEngine.VB:
        result = runner.variational_bayes(output_samples=1000, **kwargs)
    elif engine == StanResultEngine.PATHFINDER:
        result = runner.pathfinder(output_samples=1000, **kwargs)
    else:
        runner.messages["error"] = f"Unknown runner engine: {engine}"
        return InferenceResult(
            None, runner.messages, runtime=(dt.datetime.now() - time_start)
        )
    assert isinstance(result, InferenceResult)
    # print(messages)
    out = result.get_serializable_version(
        output_scope=output_scope,
        compress_values_with_errors=compress_values_with_errors,
    )
    return out


@broker.task
async def compute_model(
    model_code: str,
    data: dict[str, Any],
    model_name: str,
    engine: StanResultEngine,
    output_scope: StanOutputScope,
    compress_values_with_errors: bool,
    context: Annotated[Context, TaskiqDepends()],
) -> Optional[str]:
    if context is None:
        import socket

        worker_tag = socket.gethostname()
    else:
        worker_tag = context.state.worker_tag

    return serial_compute_model(
        model_code=model_code,
        data=data,
        model_name=model_name,
        engine=engine,
        output_scope=output_scope,
        compress_values_with_errors=compress_values_with_errors,
        worker_tag=worker_tag,
    )
