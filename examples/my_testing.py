from __future__ import annotations


from collections.abc import Mapping, Sequence
import os
from typing import Any

import gymnasium as gym
import numpy as np
from tqdm.auto import tqdm


import datetime
from sustaingym.envs.electricitymarket import ElectricityMarketEnv


def run_random(seeds: Sequence[int], env: gym.Env, discrete: bool) -> dict[str, np.ndarray]:
    num_eps = len(seeds)
    rewards = np.zeros((num_eps, env.T))
    energy = np.zeros((num_eps, env.T))
    prices = np.zeros((num_eps, env.T))

    if discrete:
        actions = np.zeros((num_eps, env.T), dtype=np.int32)
    else:
        actions = np.zeros((num_eps, env.T, 2))

    for ep, seed in tqdm(enumerate(seeds)):
        print("ep: ", ep)
        obs, info = env.reset(seed=seed)
        prices[ep, 0] = obs['prices previous'][0]
        energy[ep, 0] = obs['soc'][0]
        np.random.seed(seed)
        for i in range(1, env.T):
            action = np.random.uniform(low=-env.max_cost, high=env.max_cost, size=env.action_space.shape)
            obs, reward, _, _, _ = env.step(action)
            # print("random reward: ", reward)
            rewards[ep, i] = reward
            energy[ep, i] = obs['soc'][0]
            prices[ep, i] = obs['prices previous'][0]
            
            if discrete:
                actions[ep, i] = action
            else:
                actions[ep, i] = action[:, 0, 0]
    
    return {
        'rewards': rewards,
        'prices': prices,
        'energy': energy,
        'actions': actions
    }


def main():
    # dataset has 2021 data only
    env = ElectricityMarketEnv([datetime.date(2020,1,2), datetime.date(2020,3,1)])
    # obs, _ = env.reset()
    # print(obs.keys())
    run_random([1],env,False)

    print("OK")


if __name__ == "__main__":
    main()
