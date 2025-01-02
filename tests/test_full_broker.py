from StanTasq import MyTask, broker
import asyncio


async def test_task():
    print(f"Task: {MyTask.add_one}")
    await broker.startup()
    get_task = await MyTask.add_one.kiq(10)
    get_result = await get_task.wait_result(timeout=2)
    assert get_result.return_value == 11


if __name__ == "__main__":
    asyncio.run(test_task())
