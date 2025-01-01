import os

from taskiq import InMemoryBroker
from taskiq_nats import PullBasedJetStreamBroker
from taskiq_nats.result_backend import NATSObjectStoreResultBackend

env = os.environ.get("ENVIRONMENT")

result_backend = NATSObjectStoreResultBackend(
    servers="localhost",
)
broker = PullBasedJetStreamBroker(
    servers="localhost",
).with_result_backend(
    result_backend=result_backend,
)

if env and env == "pytest":
    broker = InMemoryBroker()
