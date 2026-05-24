"""
Main entry point for Sim2Real Train framework.
"""

import distutils.util
import os
import subprocess
from mujoco import egl
import torch
import mujoco
import itertools
import time
from typing import Callable, List, NamedTuple, Optional, Union
import numpy as np
from datetime import datetime
import functools
import os
from typing import Any, Dict, Sequence, Tuple, Union
from brax import base
from brax import envs
from brax import math
from brax.base import Base, Motion, Transform
from brax.base import State as PipelineState
from brax.envs.base import Env, PipelineEnv, State
from brax.io import html, mjcf, model
from brax.mjx.base import State as MjxState
from brax.training.agents.ppo import networks as ppo_networks
from brax.training.agents.ppo import train as ppo
from brax.training.agents.sac import networks as sac_networks
from brax.training.agents.sac import train as sac
from etils import epath
from flax import struct
from flax.training import orbax_utils
from IPython.display import HTML, clear_output
import jax
from jax import numpy as jp
from matplotlib import pyplot as plt
import mediapy as media
from ml_collections import config_dict
import mujoco
from mujoco import mjx
import numpy as np
from orbax import checkpoint as ocp
import pickle

# Project imports
import robot
from playground import wrapper
from playground import brax_ppo_params


def configure_mujoco():
    # Tell XLA to use Triton GEMM, this improves steps/sec by ~30% on some GPUs
    xla_flags = os.environ.get("XLA_FLAGS", "")
    xla_flags += " --xla_gpu_triton_gemm_any=True"
    os.environ["XLA_FLAGS"] = xla_flags

    # Set high precision for smaller GPU's (see mujoco-playground documentation)
    jax.config.update("jax_default_matmul_precision", "highest")

    # Set this to True to disable JIT compilation to enable debugging (slow)
    jax.config.update("jax_disable_jit", False)

    # Configure MuJoCo to use the EGL rendering backend (requires GPU)
    os.environ["MUJOCO_GL"] = "egl"


# Progress tracking for training curves
x_data, y_data, y_dataerr = [], [], []
times = [datetime.now()]


def progress(num_steps, metrics):
    times.append(datetime.now())
    print(
        f"Step {num_steps}: reward={metrics['eval/episode_reward']:.3f} +/- {metrics['eval/episode_reward_std']:.3f}"
    )


def main():
    debug = False
    policies_dir = epath.Path("policies")
    policies_dir.mkdir(parents=True, exist_ok=True)

    configure_mujoco()

    print(f"Default GPU: {torch.cuda.get_device_name(torch.cuda.current_device())}")

    # empty policies folder
    for file in policies_dir.iterdir():
        if file.is_file():
            file.unlink()

    env_cfg = robot.default_config()
    print(f"Environment config:\n{env_cfg}")

    env = robot.Robot(env_cfg)

    # Set PPO Params
    ppo_params = brax_ppo_params.brax_ppo_config(env_cfg.episode_length)

    # Very fast parameters for debugging, not meant to produce good results, just to check that the training loop runs.
    if debug:
        ppo_params = brax_ppo_params.brax_ppo_config_debug()

    print(f"PPO training config:\n{ppo_params}")

    # Run Training
    ppo_training_params = dict(ppo_params)
    network_factory = ppo_networks.make_ppo_networks
    if "network_factory" in ppo_params:
        del ppo_training_params["network_factory"]
        network_factory = functools.partial(
            ppo_networks.make_ppo_networks, **ppo_params.network_factory
        )

    train_fn = functools.partial(
        ppo.train,
        **dict(ppo_training_params),
        network_factory=network_factory,
        progress_fn=progress,
    )

    make_inference_fn, params, metrics = train_fn(
        environment=env,
        wrap_env_fn=wrapper.wrap_for_brax_training,
    )

    print(f"time to jit: {times[1] - times[0]}")
    print(f"time to train: {times[-1] - times[1]}")

    # -----------------------------
    # 🔬 Save trained model
    # -----------------------------
    # Save the trained policy and training metrics
    policy_file_name = policies_dir / f"robot_policy.pkl"

    with open(f"{policy_file_name}", "wb") as f:
        pickle.dump(params, f)

    # Save video of trained policy

    # reload params to ensure they were saved correctly (and to demonstrate loading)
    with open(f"{policy_file_name}", "rb") as f:
        loaded_params = pickle.load(f)

    print(f"Rendering trained policy video")
    jit_reset = jax.jit(env.reset)
    jit_step = jax.jit(env.step)
    jit_inference_fn = jax.jit(make_inference_fn(loaded_params, deterministic=True))

    rng = jax.random.PRNGKey(42)
    rollout = []
    n_episodes = 1
    frames_to_render = 150
    for _ in range(n_episodes):
        state = jit_reset(rng)
        rollout.append(state)
        for i in range(frames_to_render):
            act_rng, rng = jax.random.split(rng)
            ctrl, _ = jit_inference_fn(state.obs, act_rng)
            state = jit_step(state, ctrl)
            rollout.append(state)

    frames = env.render(rollout, camera="cam0", height=480, width=640)

    video_path = policies_dir / f"robot_policy.mp4"

    media.write_video(video_path.as_posix(), frames, fps=1.0 / env.dt)


if __name__ == "__main__":
    main()
