# Copyright 2025 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""RL config for DM Control Suite."""

from typing import Optional
from ml_collections import config_dict


def brax_ppo_config(episode_length) -> config_dict.ConfigDict:
    """Returns tuned Brax PPO config"""

    rl_config = config_dict.create(
        num_timesteps=60_000_000,
        num_evals=10,
        reward_scaling=1.0,
        episode_length=episode_length,
        normalize_observations=False,
        action_repeat=1,
        unroll_length=20,
        num_minibatches=32,
        num_updates_per_batch=8,
        discounting=0.995,
        learning_rate=0.0003,
        entropy_cost=3e-3,
        num_envs=4096,
        batch_size=4096,
        num_resets_per_eval=10,
    )

    return rl_config


def brax_ppo_config_debug() -> config_dict.ConfigDict:
    """Returns Brax PPO for fast debugging config"""

    rl_config = config_dict.create(
        num_timesteps=50000,
        num_evals=1,
        reward_scaling=1,
        episode_length=200,
        normalize_observations=True,
        action_repeat=1,
        unroll_length=5,
        num_minibatches=1,
        num_updates_per_batch=1,
        discounting=0.995,
        learning_rate=1e-3,
        entropy_cost=0,
        num_envs=64,
        batch_size=256,
        num_resets_per_eval=1,
    )
    return rl_config
