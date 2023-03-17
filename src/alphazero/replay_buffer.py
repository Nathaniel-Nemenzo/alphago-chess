import time
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
        """Samples batch_size number of examples from this replay buffer instance. If the batch_size specified is > len(buffer), then we will wait IN THIS CLASS until there are enough elements to satisfy the request.

        Args:
            batch_size (int): Number of elements to sample

        Returns:
            list[tuple]: Returns a list of sampled training examples
        """
        while batch_size > self.__len__():
            time.sleep(30) # Sleep for 30 seconds and then check again
        with self.lock:
            samples = random.sample(self.buffer, batch_size)
        return samples

    def __len__(self) -> int:
        with self.lock:
            return len(self.buffer)
