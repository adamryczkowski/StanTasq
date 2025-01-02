import os

from taskiq import InMemoryBroker
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

if env and env == "pytest":
    broker = InMemoryBroker()
