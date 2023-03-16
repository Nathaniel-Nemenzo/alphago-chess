import random
import threading

from collections import deque

class ReplayBuffer:
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.buffer = deque(maxlen = capacity)
        self.lock = threading.Lock()

    def add(self, record: tuple) -> None:
        with self.lock:
            self.buffer.append(record)

    def extend(self, items: list[tuple]):
        with self.lock:
            self.buffer.extend(items)

    def sample(self, batch_size: int) -> list[tuple]:
        with self.lock:
            samples = random.sample(self.buffer, batch_size)
        return samples

    def __len__(self) -> int:
        with self.lock:
            return len(self.buffer)
