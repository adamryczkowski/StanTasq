from StanTasq import (
    StanResultEngine,
    StanResultMainEffects,
)
from ValueWithError import make_ValueWithError


def test_MainEffects():
    one_dim_pars = {"par": make_ValueWithError(10, 1, 100)}
    main_effects = StanResultMainEffects(
        one_dim_pars=one_dim_pars,
        par_dimensions={"par": [1]},
        method_name=StanResultEngine.LAPLACE,
        calculation_sample_count=100,
    )
    dump_data = main_effects.model_dump_json()

    print(dump_data)

    main_effects2 = StanResultMainEffects.model_validate_json(dump_data)
    print(main_effects2)


if __name__ == "__main__":
    test_MainEffects()
