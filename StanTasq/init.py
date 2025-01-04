from .broker import broker
from .all_tasks import all_tasks


async def init():
    await broker.startup()
    await all_tasks.startup()
