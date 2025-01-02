from StanTasq import AllTasks
import asyncio


async def test_all_tasks():
    all_tasks = AllTasks()
    await all_tasks.startup()
    os = all_tasks.object_store
    objects = [a for a in await os.list()]
    for object in objects:
        print(object.bucket)
        value = await os.get(name=object.name)
        print(value)


if __name__ == "__main__":
    asyncio.run(test_all_tasks(), debug=True)
