from StanTasq import (
    StanTask,
    broker,
    StanResultEngine,
    StanOutputScope,
    StanResultMainEffects,
)
import asyncio
import pytest


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


@pytest.mark.asyncio
async def test_task():
    print(f"Task: {StanTask.compute_model}")
    await broker.startup()
    get_task = await StanTask.compute_model.kiq(
        model_code=model(),
        data={"arr": 1.0},
        model_name="test1",
        engine=StanResultEngine.LAPLACE,
        output_scope=StanOutputScope.MainEffects,
        compress_values_with_errors=True,
    )
    get_result = await get_task.wait_result(timeout=200)
    result = get_result.return_value
    if result is None:
        print(f"Error: {get_result.error}")
    obj = StanResultMainEffects(**result)
    print(result)
    print(obj)
    # stream = await broker.js.stream_info("stan_tasks")


if __name__ == "__main__":
    # asyncio.run(test_task())
    asyncio.run(test_task())
