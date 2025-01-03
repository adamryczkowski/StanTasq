from taskiq import AsyncTaskiqTask

from StanTasq import AllTasks, broker, MyTask
import asyncio


async def clear_all():
    all_tasks = AllTasks()
    await all_tasks.startup()
    print("Deleting all tasks:")
    print(all_tasks)
    await all_tasks.clear_all_tasks()
    await broker.shutdown()
    await broker.startup()


async def test_clear():
    await clear_all()
    get_task: AsyncTaskiqTask = await MyTask.add_one.kiq(10)
    all_tasks = AllTasks()
    await all_tasks.startup()
    task = all_tasks.get_task(get_task.task_id)
    print(task)
    result = await task.wait_result()
    assert result.return_value == 11
    get_result = await get_task.wait_result(timeout=20)
    assert get_result.return_value == 11
    print(task)


async def all_tests():
    await broker.startup()
    await test_clear()


if __name__ == "__main__":
    # asyncio.run(test_task())
    asyncio.run(all_tests(), debug=True)
