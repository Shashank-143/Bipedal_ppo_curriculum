import numpy as np
import torch

REWARD_SCALE = 0.1

class PPO:
    def __init__(self, model, lr= 1e-4, gamma=0.99, lam=0.95, 
                 clip=0.2, epochs=10, batch_size=64):
        self.model = model
        self.gamma = gamma
        self.lam = lam
        self.clip = clip
        self.epochs = epochs
        self.batch_size = batch_size
        self.optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        self.episode_score = 0.0
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        def collect(self, env, obs, normalizer, n_steps=2048):
            """
            Performs n_steps of interaction with the environment, collecting observations, actions, rewards, and values.
            """
            states, actions, rewards, dones, values, log_probs = [], [], [], [], [], []
            finished_scores = []

            for _ in range(n_steps):
                normalizer.update(obs)
                state = torch.as_tensor(normalizer.normalize(obs), dtype=torch.float32).to(self.device)

                with torch.no_grad():
                    dist = self.model.action_distribution(state)
                    action = dist.sample()
                    log_prob = dist.log_prob(action).sum()
                    value = self.model.value(state)

                # Torque acceptable range is [-1, 1]
                next_obs, reward, terminated, truncated, info = env.step(np.clip(action.cpu().numpy(), -1, 1))
                done = terminated or truncated

                states.append(state.cpu().numpy())
                actions.append(action.cpu().numpy())
                log_probs.append(log_prob.cpu().numpy())
                rewards.append(reward * REWARD_SCALE)
                dones.append(done)
                values.append(value.cpu().numpy())

                self.episode_score += info["raw_reward"]
                if done:
                    finished_scores.append(self.episode_score)
                    self.episode_score = 0.0
                    obs, _ = env.reset()

                else:
                    obs = next_obs

            last_stage = torch.as_tensor(normalizer.normalize(obs), dtype=torch.float32).to(self.device)
            with torch.no_grad():
                last_value = self.model.value(last_stage).cpu().numpy()

            advantages, returns = self.compute_advantages(rewards, dones, values, last_value)
            batch = {
                "states": torch.as_tensor(np.array(states), dtype=torch.float32),
                "actions": torch.as_tensor(np.array(actions), dtype=torch.float32),
                "log_probs": torch.as_tensor(np.array(log_probs), dtype=torch.float32),
                "advantages": torch.as_tensor(advantages, dtype=torch.float32),
                "returns": torch.as_tensor(returns, dtype=torch.float32),
            }
            return obs, batch, finished_scores

    def compute_advantages(self, rewards, dones, values, last_value):
        """
        Computes GAE for each step.
        """
        n = len(rewards)
        advantages = np.zeros(n, dtype=np.float32)
        running = 0.0
        for t in reversed(range(n)):
            next_value = last_value if t == n - 1 else values[t + 1]
            not_done = 1.0 - dones[t]

            surprise = rewards[t] + self.gamma * next_value * not_done - values[t]
            running = surprise + self.gamma * self.lam * not_done * running
            advantages[t] = running
        returns = advantages + values
        return advantages, returns

    def update(self, batch):
        """
        Improve the network using the collected batch of data.
        """
        advantages = batch["advantages"].to(self.device)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        n = len(advantages)

        for _ in range(self.epochs):
            order = torch.randperm(n)
            for start in range(0, n, self.batch_size):
                idx = order[start:start + self.batch_size]

                new_log_prob, entropy, value = self.model.evaluate(
                    batch["states"][idx].to(self.device), 
                    batch["actions"][idx].to(self.device))

                ratio = (new_log_prob - batch["log_probs"][idx].to(self.device)).exp()
                adv = advantages[idx]

                unclipped = ratio * adv
                clipped = torch.clamp(ratio, 1 - self.clip, 1 + self.clip) * adv
                policy_loss = -torch.min(unclipped, clipped).mean()
                value_loss = ((value - batch["returns"][idx].to(self.device)) ** 2).mean()

                loss = policy_loss + 0.5 * value_loss - 0.01 * entropy.mean()
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)
                self.optimizer.step()