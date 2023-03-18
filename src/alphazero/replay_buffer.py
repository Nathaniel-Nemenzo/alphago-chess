import time
import random

class ReplayBuffer:
    def __init__(self, buffer, capacity: int):
        self.buffer = buffer
        self.capacity = capacity

    def add(self, record: tuple) -> None:
        self.buffer.put(record)

    def extend(self, items: list[tuple]):
        for item in items:
            self.buffer.put(item)

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
        return len(self.buffer)
