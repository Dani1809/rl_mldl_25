import gym
import argparse
import numpy as np
from env.custom_hopper import *
from stable_baselines3 import PPO, SAC

def make_env(domain="target", distribution="uniform"):
    return gym.make(f'CustomHopper-{domain}-v0', distribution=distribution)

def train_and_evaluate(args):
    # Creazione degli ambienti separati per il training e il testing
    test_env = make_env(domain=args.test_domain, distribution="uniform")  # test standardizzato
    params = [0.1, 0.3, 0.5, 0.7, 1.0]
    # Training
    for param in params:
        print(f"\n=== Training with param={param} ===")
        model_name = f"{args.algo}_{args.train_domain}_{args.distribution}_param{param}"
        train_env = make_env(domain=args.train_domain, distribution=args.distribution, param=param)

        # TRAINING
        if args.algo == "ppo":
            if args.train:
                model = PPO("MlpPolicy", train_env, verbose=0)
                model.learn(total_timesteps=args.timesteps)
                model.save(model_name)
            else:
                model = PPO.load(model_name)
        elif args.algo == "sac":
            if args.train:
                model = SAC("MlpPolicy", train_env, verbose=0)
                model.learn(total_timesteps=args.timesteps)
                model.save(model_name)
            else:
                model = SAC.load(model_name)

        # EVALUATION
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

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train & evaluate PPO/SAC on CustomHopper")
    parser.add_argument("--algo", type=str, default="ppo", choices=["ppo", "sac"], help="RL algorithm to use")
    parser.add_argument("--train", action="store_true", help="Train a new model")
    parser.add_argument("--train_domain", type=str, default="source", choices=["source", "target"], help="Env domain for training")
    parser.add_argument("--test_domain", type=str, default="target", choices=["source", "target"], help="Env domain for testing")
    parser.add_argument("--distribution", type=str, default="uniform", choices=["uniform", "normal", "lognormal"], help="Mass distribution")
    parser.add_argument("--timesteps", type=int, default=1000000, help="Number of training timesteps")
    parser.add_argument("--episodes", type=int, default=50, help="Number of evaluation episodes")

    args = parser.parse_args()
    train_and_evaluate(args)