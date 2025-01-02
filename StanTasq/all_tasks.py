import datetime as dt

import humanize
from nats.aio.msg import Msg
from nats.js import JetStreamContext
from taskiq.result import TaskiqResult
from taskiq.task import AsyncTaskiqTask
from tqdm import tqdm
from taskiq_nats.result_backend import NATSObjectStoreResultBackend


from .broker import broker


class TaskWrapper:
    _task: AsyncTaskiqTask
    _metadata: Msg.Metadata
    _name: str
    _args: list
    _kwargs: dict
    _result: None | TaskiqResult

    def __init__(
        self,
        task_id: str,
        metadata: Msg.Metadata,
        task_name: str,
        args: list,
        kwargs: dict,
    ):
        self._task = AsyncTaskiqTask(
            task_id=task_id, result_backend=broker.result_backend
        )
        self._metadata = metadata
        self._name = task_name
        self._args = args
        self._kwargs = kwargs

    @property
    def task(self) -> AsyncTaskiqTask:
        return self._task

    @property
    def timestamp(self) -> dt.datetime:
        return self._metadata.timestamp

    @property
    def name(self) -> str:
        return self._name

    @property
    def args(self) -> list:
        return self._args

    @property
    def kwargs(self) -> dict:
        return self._kwargs

    @property
    def consumer_id(self) -> str:
        return self._metadata.consumer

    @property
    def task_id(self) -> str:
        return self._task.task_id

    async def delete_task(self):
        await self.delete_result()
        await broker.js.delete_message(
            self._metadata.stream, self._metadata.SequencePair
        )

    async def delete_result(self):
        result_backend: NATSObjectStoreResultBackend = broker.result_backend
        await result_backend.object_store.delete(self._task.task_id)

    async def get_result(self) -> TaskiqResult:
        if self._result is None:
            self._result = await self._task.get_result(with_logs=True)
        return self._result

    def __repr__(self) -> str:
        args = ", ".join(self._args)
        kwargs = ", ".join([f"{key}={value}" for key, value in self._kwargs.items()])
        args_str = "("
        if len(kwargs) > 0:
            args_str += f"{args}, {kwargs}"
        else:
            args_str += f"{args}"
        args_str += ")"
        if self._result is None:
            ans = f"Task: {self._name}{args_str} scheduled at {self.timestamp.strftime("%a, %m.%b.%Y %H:%M:%S")}"
        else:
            ans = f"Task: {self._name}{args_str}->{self._result.return_value} executed at {self.timestamp.strftime("%a, %m.%b.%Y %H:%M:%S")} for {humanize.naturaldelta(dt.timedelta(seconds=self._result.execution_time))}"
        return ans


class AllTasks:
    _all_tasks: dict[str, TaskWrapper] | None

    async def startup(self):
        stream_info = await broker.js.stream_info(
            broker.stream_name, subjects_filter=broker.subject
        )
        sub: JetStreamContext.PullSubscription = await broker.js.pull_subscribe(
            subject=broker.stream_name, stream=broker.subject
        )
        count = stream_info.state.messages
        self._all_tasks = {}
        with tqdm(total=count, desc="Loading tasks") as pbar:
            while count > 0:
                msg = await sub.fetch(1)[0]
                count -= 1
                data = broker.serializer.loadb(msg.data)
                taskw = TaskWrapper(
                    task_id=data["task_id"],
                    metadata=msg.metadata,
                    task_name=data["task_name"],
                    args=data["args"],
                    kwargs=data["kwargs"],
                )
                await taskw.get_result()
                self._all_tasks[data["task_id"]] = taskw
                pbar.update(1)

    def __repr__(self) -> str:
        ans_list = []
        for task in self._all_tasks.values():
            ans_list.append(f"{task.task_id}: {repr(task)}")
        return "\n".join(ans_list)

    async def delete_tasks_older_than(
        self, interval: dt.timedelta = dt.timedelta(days=7)
    ):
        for task in self._all_tasks.values():
            if dt.datetime.now() - task.timestamp > interval:
                await task.delete_task()
                del self._all_tasks[task.task_id]

    async def clear_all_tasks(self):
        """VERY DANGEROUS! Deletes all messages and results without any confirmation and possibility to restore."""
        await broker.js.delete_stream(broker.stream_name)
        result_backend: NATSObjectStoreResultBackend = broker.result_backend
        await broker.js.delete_object_store(result_backend.object_store.name)
