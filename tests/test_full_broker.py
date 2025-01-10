from StanTasq import MyTask, broker
import asyncio
from nats.js import JetStreamContext
from taskiq.task import AsyncTaskiqTask  # , TaskiqResult


async def test_task():
    print(f"Task: {MyTask.add_one}")
    await broker.startup()
    get_task = await MyTask.add_one.kiq(10)
    get_result = await get_task.wait_result(timeout=20)
    assert get_result.return_value == 11
    # stream = await broker.js.stream_info("stan_tasks")


async def test_task_pull():
    await broker.startup()
    stream_info = await broker.js.stream_info(
        "stan_tasks", subjects_filter="stan_tasks"
    )
    sub: JetStreamContext.PullSubscription = await broker.js.pull_subscribe(
        subject="stan_tasks", stream="stan_tasks"
    )
    count = stream_info.state.messages
    while count > 0:
        msg = await sub.fetch(1)
        print(msg[0].metadata.timestamp)
        print(msg[0])
        count -= 1
        data = broker.serializer.loadb(msg[0].data)
        task = AsyncTaskiqTask(
            task_id=data["task_id"], result_backend=broker.result_backend
        )
        assert await task.is_ready()
        print(f"Args: {data["args"]}")
        result = await task.get_result(with_logs=True)
        print(f"Result: {result.return_value}")
        print(f"Execution time: {result.execution_time}")
        print(f"Logs: {result.log}")
        print()


if __name__ == "__main__":
    # asyncio.run(test_task())
    asyncio.run(test_task_pull(), debug=True)
