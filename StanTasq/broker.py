import os

from taskiq import InMemoryBroker, TaskiqEvents, TaskiqState
from taskiq.serializers import JSONSerializer
from taskiq_nats import PullBasedJetStreamBroker
from taskiq_nats.result_backend import NATSObjectStoreResultBackend

env = os.environ.get("ENVIRONMENT")

result_backend = NATSObjectStoreResultBackend(
    servers="localhost",
    token="szakal",
    keep_results=True,
    bucket_name="stan_bucket",
    serializer=JSONSerializer(),
)

broker = PullBasedJetStreamBroker(
    servers="localhost", stream_name="stan_tasks", token="szakal", subject="stan_tasks"
).with_result_backend(
    result_backend=result_backend,
)


@broker.on_event(TaskiqEvents.WORKER_STARTUP)
async def startup(state: TaskiqState) -> None:
    # Here we store connection pool on startup for later use.
    import socket

    state.worker_tag = socket.gethostname()


if env and env == "pytest":
    broker = InMemoryBroker()
