import torch
import ray

# TASK: test this

ray.init()

@ray.remote
class MCTS:
    def __init__(
            self,
            game: any,
            model: torch.nn.Module,
            epsilon_fix: bool = True,
    ):
        self.game = game
        self.model = model
        self.tree = {}
        self.epsilon_fix = epsilon_fix

    def simulate(
            self,
            state: any,
            cpuct: float = 1.0,
    ):
        path = []
        current_player = self.game.get_player(state)

        while True:
            hashed_s = str(state)
            if hashed_s in self.tree: # Not at leaf; select
                stats = self.tree[hashed_s]
                N, Q, P = stats[:, 1], stats[:, 2], stats[:, 3]
                U = cpuct * P * torch.sqrt(torch.sum(N) + 1e-6 if self.epsilon_fix else 0) / (1 + N)
                best_a_idx = torch.argmax(Q + U)

                # Pick the best action
                best_a = stats[best_a_idx, 0]
                template = torch.zeros_like(self.game.get_valid_actions(state))
                template[tuple(best_a)] = 1
                s_prime = self.game.take_action(state, template)
                path.append((hashed_s, best_a_idx))
                state = s_prime 
            else:
                break

        w = self.game.check_winner(state)
        # TODO: fix this
        if w is not None:
            v = 1 if w is not -1 else 0
        else: # At leaf; expand
            valid_actions = self.game.get_valid_actions(state)
            idx = torch.stack(torch.where(valid_actions == 1)).T
            p, v = self.model(state)
            stats = torch.zeros((len(valid_actions), 4))
            stats[:, 0] = idx[:, 0]
            stats[:, -1] = p
            self.tree[hashed_s] = stats

        winning_player = w if w is not None else current_player

        # Update the visited nodes in the path
        for hashed_s, best_a_idx in reversed(path):
            stats = self.tree[hashed_s]
            n, q = stats[best_a_idx, 1], stats[best_a_idx, 2]
            adj_v = v if current_player == winning_player else -v
            stats[best_a_idx, 2] = (n * q + adj_v) / (n + 1)
            stats[best_a_idx, 1] += 1
            v = adj_v

    def run(
            self,
            state: any,
            num_iterations: int,
    ):
        for _ in range(num_iterations):
            self.simulate(state)
        
        return self.tree

def get_improved_policy(
        state: any,
        num_iterations: int,
        num_workers: int,
        game: any,
        model: torch.nn.Module,
        temperature: float = 1.0,
):
    # prep
    hashed_s = str(state)

    # run mcts
    mcts_runs = [MCTS.remote(game, model).run(state, num_iterations) for _ in range(num_workers)]
    results = ray.get(mcts_runs)

    # merge
    tree = {}
    for result in results:
        stats = result[hashed_s]
        if hashed_s not in tree:
            tree[hashed_s] = stats
        else:
            # merge the visit counts
            tree[hashed_s][:, 1] += stats[:, 1]
    if len(tree) == 0:
        return None
    
    # return improved policy based on temperature
    stats = tree[hashed_s]
    N = stats[:, 1]
    if temperature == 0:
        policy = torch.zeros_like(N)
        policy[torch.argmax(N)] = 1
    else:
        policy = N ** (1 / temperature)
        policy /= torch.sum(policy)

    return policy

def get_action(
        state: any,
        num_iterations: int,
        num_workers: int,
        game: any,
        model: torch.nn.Module,
        temperature: float = 1.0,
):
    policy = get_improved_policy(
        state,
        num_iterations,
        num_workers,
        game,
        model,
        temperature,
    )
    if policy is None:
        return None
    return torch.multinomial(policy, 1).item()