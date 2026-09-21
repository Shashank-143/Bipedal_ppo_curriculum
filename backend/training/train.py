import csv
import random
from pathlib import Path
import numpy as np
import torch
from env.wrappers import make_env
from models.actor_critic import ActorCritic, Normalizer, save_model
from training.curriculum import Curriculum, STAGES
from training.ppo_engine import PPO

CHECKPOINT_DIR = Path("checkpoints")
STEPS_PER_UPDATE = 2048
MAX_STEPS_PER_RANGE = 300_000

def train(total_steps=1_000_000, learning_rate=3e-4, seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    curriculum = Curriculum(STAGES)
    model = ActorCritic()
    normalizer = Normalizer()
    ppo = PPO(model, lr=learning_rate)

    env = make_env(**curriculum.env_settings())
    obs, _ = env.reset()

    log_name = "curriculum_training_log.csv"
    log_file = open(CHECKPOINT_DIR / log_name, "w", newline="")
    log = csv.writer(log_file)
    log.writerow(["step", "stage", "average_score", "episode_score"])
    steps, steps_in_stage, update_number = 0, 0, 0

    while steps < total_steps:
        obs, batch, finished_scores = ppo.collect(env, obs, normalizer, n_steps=STEPS_PER_UPDATE)
        ppo.update(batch)

        steps += STEPS_PER_UPDATE
        steps_in_stage += STEPS_PER_UPDATE
        update_number += 1

        curriculum.add_scores(finished_scores)
        avg_score = curriculum.average_score()
        log.writerow([steps, curriculum.stage["name"], avg_score, sum(finished_scores)])
        log_file.flush()
        if update_number % 5 == 0:
            print(f"Step: {steps:>9,}, Stage: {curriculum.stage['name']}, Average Score: {avg_score:11.1f}, Episode Score: {sum(finished_scores):7.1f}")

        is_last_stage = curriculum.index == len(curriculum.stages) - 1
        stuck_in_stage = steps_in_stage >= MAX_STEPS_PER_RANGE and not is_last_stage
        if is_last_stage and stuck_in_stage:
            print(f"Training stopped after {steps:,} steps. Reached last stage and stuck in it.")
            break

        if curriculum.ready_to_promote():
            finished = curriculum.index + 1
            save_model(CHECKPOINT_DIR / f"model_stage_{finished}.pth", model, normalizer)
            print(f"Finished stage {finished}. Promoting to stage {curriculum.index + 2}.")
            print(f"Saving model to {CHECKPOINT_DIR / f'model_stage_{finished}.pth'}")
            curriculum.promote()
            print(f"Promoted to stage {curriculum.index + 1}: {curriculum.stage['name']}")

            env.close()
            env = make_env(**curriculum.env_settings())
            obs, _ = env.reset()
            ppo.episode_score, steps_in_stage = 0.0, 0

    final_name = "model_stage_final.pth"
    save_model(CHECKPOINT_DIR / final_name, model, normalizer)
    log_file.close()
    print(f"Training completed after {steps:,} steps. Saving final model to {CHECKPOINT_DIR / final_name}")