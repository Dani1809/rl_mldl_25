import gym
import argparse
import numpy as np
import random
import torch
from env.custom_hopper import *
from stable_baselines3 import PPO

SEED = 42

def set_global_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

def make_env(domain="target", distribution="uniform", param=None):
    env = gym.make(f'CustomHopper-{domain}-v0', distribution=distribution, param=param)
    env.seed(SEED)
    env.action_space.seed(SEED)
    env.observation_space.seed(SEED)
    return env

def train_and_evaluate(args):
    set_global_seed(SEED)
    test_env = make_env(domain=args.test_domain, distribution="uniform")
    params = [0.1, 0.3, 0.5, 0.7, 1.0]
    learning_rate=0.001
    clip_range=0.2
    n_steps=4096
    results_file = open(f"{args.train_domain}_{args.distribution}_{learning_rate}_{clip_range}_{n_steps}.txt", "w")
    results_file.write("Param\tMean Reward\tStd Dev\n")

    for param in params:
        print(f"\n=== Training with param={param} ===")
        model_name = f"ppo_{args.train_domain}_{args.distribution}_param{param}"
        train_env = make_env(domain=args.train_domain, distribution=args.distribution, param=param)

        if args.train:
            model = PPO("MlpPolicy", train_env, learning_rate=learning_rate, clip_range=clip_range, n_steps=n_steps, verbose=0, seed=SEED)
            model.learn(total_timesteps=args.timesteps)
            model.save(model_name)
        else:
            model = PPO.load(model_name)

        rewards = []
        obs = test_env.reset()
        cumulative_reward = 0
        i = 0

        while i < args.episodes:
            action, _ = model.predict(obs)
            obs, reward, done, _ = test_env.step(action)
            cumulative_reward += reward
            if done:
                print(f"[{i+1}] Ricompensa cumulativa: {cumulative_reward:.2f}")
                rewards.append(cumulative_reward)
                cumulative_reward = 0
                obs = test_env.reset()
                i += 1

        print(f"→ Param: {param} | Media: {np.mean(rewards):.2f}, Dev Std: {np.std(rewards):.2f}")
        results_file.write(f"{param}\t{np.mean(rewards):.2f}\t{np.std(rewards):.2f}\n")

    results_file.close()
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train & evaluate PPO on CustomHopper")
    parser.add_argument("--train", action="store_true", help="Train a new model")
    parser.add_argument("--train_domain", type=str, default="source", choices=["source", "target"], help="Env domain for training")
    parser.add_argument("--test_domain", type=str, default="target", choices=["source", "target"], help="Env domain for testing")
    parser.add_argument("--distribution", type=str, default="uniform", choices=["uniform", "normal"], help="Mass distribution")
    parser.add_argument("--timesteps", type=int, default=1000000, help="Number of training timesteps")
    parser.add_argument("--episodes", type=int, default=50, help="Number of evaluation episodes")

    args = parser.parse_args()
    train_and_evaluate(args)