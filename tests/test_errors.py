from StanTasq import (
    StanTask,
    StanResultEngine,
    StanOutputScope,
)


def test_parse_error():
    bad_model = "bad model, syntax error."
    get_task = StanTask.serial_compute_model(
        model_code=bad_model,
        data={"par": 1.0},
        model_name="bad_model",
        engine=StanResultEngine.MCMC,
        output_scope=StanOutputScope.MainEffects,
        compress_values_with_errors=True,
        worker_tag="local",
        require_converged=False,
    )
    print(repr(get_task))


if __name__ == "__main__":
    test_parse_error()
