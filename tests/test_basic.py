from StanTasq import add_one
import pytest


@pytest.mark.anyio
async def test_task():
    assert await add_one(10) == 11
