from StanTasq import (
    StanTask,
    StanResultEngine,
    StanOutputScope,
)


def model() -> str:
    return """data {
  real arr;
}
parameters {
  real par;
}
model {
  arr ~ normal(par, 1);
}
generated quantities {
  real par3;
  par3 = par + 1;
}
"""


def test_task():
    print(f"Task: {StanTask.compute_model}")
    data = {"arr": 1.0}
    get_task = StanTask.serial_compute_model(
        model_code=model(),
        data=data,
        model_name="test1",
        engine=StanResultEngine.VB,
        output_scope=StanOutputScope.MainEffects,
        compress_values_with_errors=True,
        worker_tag="local",
        require_converged=False,
    )
    print(repr(get_task))
    get_task = StanTask.serial_compute_model(
        model_code=model(),
        data=data,
        model_name="test1",
        engine=StanResultEngine.LAPLACE,
        output_scope=StanOutputScope.MainEffects,
        compress_values_with_errors=False,
        worker_tag="local",
    )
    print(repr(get_task))
    get_task = StanTask.serial_compute_model(
        model_code=model(),
        data=data,
        model_name="test1",
        engine=StanResultEngine.MCMC,
        output_scope=StanOutputScope.MainEffects,
        compress_values_with_errors=True,
        worker_tag="local",
    )
    print(repr(get_task))
    get_task = StanTask.serial_compute_model(
        model_code=model(),
        data=data,
        model_name="test1",
        engine=StanResultEngine.PATHFINDER,
        output_scope=StanOutputScope.MainEffects,
        compress_values_with_errors=True,
        worker_tag="local",
    )
    print(repr(get_task))


if __name__ == "__main__":
    # asyncio.run(test_task())
    test_task()
