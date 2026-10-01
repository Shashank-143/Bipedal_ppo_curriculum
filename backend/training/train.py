import csv
import random
from pathlib import Path
import numpy as np
import torch
from env.wrappers import make_env, run_episode
from models.actor_critic import ActorCritic, Normalizer, save_model, load_policy
from training.curriculum import Curriculum, STAGES, WINDOW
from training.ppo_engine import PPO

CHECKPOINT_DIR = Path("checkpoints")
STEPS_PER_UPDATE = 2048
MAX_STEPS_PER_RANGE = 300_000
EVAL_SEEDS = range(100, 110)
SUMMARY_FILE = CHECKPOINT_DIR / "best_models.txt"

def evaluate(path, settings):
    """Average score of a saved checkpoint over fixed seeds, on the stage's own settings."""
    policy = load_policy(path)
    scores = [run_episode(policy, seed, **settings)["score"] for seed in EVAL_SEEDS]
    return sum(scores) / len(scores)

def finish_stage(number, name, settings, model, normalizer, best_score):
    """Saves the stage's final model, compares it with the stage's best, and records the winner."""
    final_path = CHECKPOINT_DIR / f"model_stage_{number}_final.pth"
    best_path = CHECKPOINT_DIR / f"model_stage_{number}_best.pth"
    save_model(final_path, model, normalizer)

    if best_score is None:
        line = f"{name}: no best model (score window never filled), winner=final ({final_path.name})"
    else:
        best_eval = evaluate(best_path, settings)
        final_eval = evaluate(final_path, settings)
        winner_path = best_path if best_eval >= final_eval else final_path
        winner = "best" if winner_path == best_path else "final"
        line = (f"{name}: best={best_eval:.1f}, final={final_eval:.1f}, "
                f"winner={winner} ({winner_path.name})")

    print(line)
    with open(SUMMARY_FILE, "a") as f:
        f.write(line + "\n")

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

    open(SUMMARY_FILE, "w").close()
    log_name = "curriculum_training_log.csv"
    log_file = open(CHECKPOINT_DIR / log_name, "w", newline="")
    log = csv.writer(log_file)
    log.writerow(["step", "stage", "average_score", "episode_score"])
    steps, steps_in_stage, update_number = 0, 0, 0
    best_score = None

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

        if len(curriculum.recent_scores) >= WINDOW and (best_score is None or avg_score > best_score):
            best_score = avg_score
            best_path = CHECKPOINT_DIR / f"model_stage_{curriculum.index + 1}_best.pth"
            save_model(best_path, model, normalizer)
            print(f"New best for {curriculum.stage['name']}: {best_score:.1f}. Saved to {best_path}")

        if update_number % 5 == 0:
            print(f"Step: {steps:>9,}, Stage: {curriculum.stage['name']}, Average Score: {avg_score:11.1f}, Episode Score: {sum(finished_scores):7.1f}")

        stuck_in_stage = steps_in_stage >= MAX_STEPS_PER_RANGE and not curriculum.ready_to_promote()
        if stuck_in_stage:
            print(f"Training stopped after {steps:,} steps. Stuck in {curriculum.stage['name']}.")
            break

        if curriculum.ready_to_promote():
            finished = curriculum.index + 1
            print(f"Finished stage {finished}. Promoting to stage {curriculum.index + 2}.")
            finish_stage(finished, curriculum.stage["name"], curriculum.env_settings(),
                         model, normalizer, best_score)
            curriculum.promote()
            print(f"Promoted to stage {curriculum.index + 1}: {curriculum.stage['name']}")

            env.close()
            env = make_env(**curriculum.env_settings())
            obs, _ = env.reset()
            ppo.episode_score, steps_in_stage, best_score = 0.0, 0, None

    # Covers the stage training ended in (stuck, or total_steps ran out).
    if steps_in_stage > 0:
        finish_stage(curriculum.index + 1, curriculum.stage["name"], curriculum.env_settings(),
                     model, normalizer, best_score)

    log_file.close()
    print(f"Training completed after {steps:,} steps. Summary saved to {SUMMARY_FILE}")