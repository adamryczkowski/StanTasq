import os

from taskiq import AsyncBroker, InMemoryBroker

env = os.environ.get("ENVIRONMENT")

broker: AsyncBroker = InMemoryBroker()  # ZeroMQBroker()

if env and env == "pytest":
    broker = InMemoryBroker()
