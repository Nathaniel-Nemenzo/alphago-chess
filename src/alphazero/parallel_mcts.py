import torch
import ray

ray.init()

class MCTSNode:
    def __init__(self, 
                 state: any,
                 parent = None,
                 action = None):
        self.state = state
        self.parent = parent
        self.action = action
        self.children = []
        self.visits = 0
        self.total_reward = 0