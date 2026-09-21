import gymnasium as gym
import numpy as np

class DifficultWrapper(gym.Wrapper):
    def __init__(self, env, gravity=1.0, friction=None, fall_penalty=1.0):
        super().__init__(env)
        self.gravity = gravity
        self.friction = friction
        self.fall_penalty = fall_penalty
        self.start_x= 0.0 

    def reset(self, **kwargs):
        self.env.unwrapped.gravity = (0, -10 * self.gravity)
        obs, info = self.env.reset(**kwargs)

        if self.friction is not None:
            for piece in self.env.unwrapped.terrain:
                for fixture in piece.fixtures:
                    fixture.friction = self.friction

        self.start_x = float(self.env.unwrapped.hull.position.x)
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        info["raw_reward"] = float(reward)
        info["fell"] = bool(terminated and reward <= -100)
        info["distance"] = float(self.env.unwrapped.hull.position.x - self.start_x)

        if info["fell"]:
            reward = -100 * self.fall_penalty
        return obs, reward, terminated, truncated, info
    
def make_env(gravity=1.0, friction=None, fall_penalty=1.0, render= False):
    env = gym.make("BipedalWalker-v3", render_mode="rgb_array" if render else None)
    return DifficultWrapper(env, gravity=gravity, friction=friction, fall_penalty=fall_penalty)

def run_episode(policy, seed, render= False, **settings):
    """
    Plays one full episode using the given policy.
    """
    env = make_env(**settings, render=render)
    obs, _ = env.reset(seed=seed)

    frames, torques =[], []
    score, steps= 0.0, 0
    while True:
        action = np.clip(policy(obs), -1, 1)
        obs, reward, terminated, truncated, info = env.step(action)

        score += info["raw_reward"]
        steps += 1
        torques.append(action.tolist())
        if render and steps % 2 == 0:
            frames.append(env.render())

        if terminated or truncated:
            break

    env.close()
    return {
        "score": score,
        "steps": steps,
        "fell": info["fell"],
        "distance": info["distance"],
        "torques": torques,
        "frames": frames
    }