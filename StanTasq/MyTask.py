from ._broker import broker
from typing import Annotated
from taskiq import Context, TaskiqDepends


@broker.task
async def add_one(value: int, context: Annotated[Context, TaskiqDepends()]) -> int:
    return value + 1
