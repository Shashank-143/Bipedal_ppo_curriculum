import numpy as np
import torch
import torch.nn as nn

class Normalizer:
    """
    Keeps a running avg of obs & rescales them to be
    mean 0 and std 1. The no.s are saved inside every checkpoint.
    """

    def __init__(self, size = 24):
        self.count = 0
        self.mean = np.zeros(size, dtype=np.float32)
        self.m2 = np.zeros(size, dtype=np.float32)

    def update(self, x):
        self.count += 1
        delta = x - self.mean
        self.mean += delta / self.count
        self.m2 += delta * (x - self.mean)

    def normalize(self, x):
        var = self.m2 / max(1, self.count)
        return np.clip((x - self.mean) / (np.sqrt(var) + 1e-8), -10, 10)


class ActorCritic(nn.Module):
    def __init__(self, obs_size=24, action_size=4, hidden=256):
        super().__init__()
        self.actor = nn.Sequential(
            nn.Linear(obs_size, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
            nn.Linear(hidden, action_size),
        )

        self.critic = nn.Sequential(
            nn.Linear(obs_size, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
            nn.Linear(hidden, 1),
        )

        # Amount of random expolaration the actor adds.
        self.log_std = nn.Parameter(torch.full((action_size,), -0.5))

    def action_distribution(self, obs):
        """
        The actor gives the avg action to take, but we add some noise to it.
        """
        mean = self.actor(obs)
        return torch.distributions.Normal(mean, self.log_std.exp())

    def value(self, obs):
        """
        The critic gives the value of the state.
        """
        return self.critic(obs).squeeze(-1)

    def evaluate(self, obs, actions):
        """
        Returns the log prob of the actions, the entropy of the distribution,
        and the value of the state.
        """
        dist = self.action_distribution(obs)
        log_prob = dist.log_prob(actions).sum(-1)
        entropy = dist.entropy().sum(-1)
        return log_prob, entropy, self.value(obs)

def save_model(path, model, normalizer):
    torch.save({
        "model" : model.state_dict(),
        "normalizer_count" : normalizer.count,
        "normalizer_mean" : normalizer.mean,
        "normalizer_m2" : normalizer.m2,
    }, path)

def load_policy(path):
    data = torch.load(path, map_location=torch.device("cpu"),weights_only=False)
    model = ActorCritic()
    model.load_state_dict(data["model"])
    model.eval()

    normalizer = Normalizer()
    normalizer.count = data["normalizer_count"]
    normalizer.mean = data["normalizer_mean"]
    normalizer.m2 = data["normalizer_m2"]

    def policy(obs):
        x = torch.as_tensor(normalizer.normalize(obs), dtype=torch.float32)
        with torch.no_grad():
            return model.actor(x).numpy()

    return policy