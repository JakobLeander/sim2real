from typing import Any, Dict, Optional, Union

from flax.nnx import data, state
import jax
import jax.numpy as jp
from jax.random import key
from ml_collections import config_dict
import mujoco
from mujoco import mjx
import logging
from etils import epath


from playground import mjx_env

log = logging.getLogger(__name__)

_XML_PATH = epath.Path(__file__).parent / "xmls/robot.xml"


def default_config() -> config_dict.ConfigDict:
    return config_dict.create(
        ctrl_dt=0.01,
        sim_dt=0.002,
        episode_length=1000,
        action_repeat=1,
        vision=False,
        impl="warp",
        naconmax=8192,
        njmax=512,
    )


class Robot(mjx_env.MjxEnv):

    def __init__(
        self,
        config: config_dict.ConfigDict = default_config(),
        config_overrides: Optional[Dict[str, Union[str, int, list[Any]]]] = None,
    ):
        super().__init__(config, config_overrides)

        if self._config.vision:
            raise NotImplementedError("Vision not implemented.")

        self._xml_path = _XML_PATH.as_posix()

        # CPU model
        self._mj_model = mujoco.MjModel.from_xml_string(_XML_PATH.read_text())
        self._mj_model.opt.timestep = self.sim_dt

        # MJX model (Warp backend)
        self._mjx_model = mjx.put_model(self._mj_model, impl=self._config.impl)

        # Build sensor lookup tables (version‑safe)
        self.sensor_adr: Dict[str, int] = {}
        self.sensor_dim: Dict[str, int] = {}

        names_buf = self._mj_model.names

        if hasattr(self._mj_model, "sensor_nameadr"):
            nameadr = self._mj_model.sensor_nameadr
        elif hasattr(self._mj_model, "name_sensoradr"):
            nameadr = self._mj_model.name_sensoradr
        else:
            raise RuntimeError("MuJoCo model has no sensor name address field.")

        for i in range(self._mj_model.nsensor):
            adr = nameadr[i]
            name_bytes = names_buf[adr:].split(b"\x00", 1)[0]
            name = name_bytes.decode("utf-8")

            self.sensor_adr[name] = self._mj_model.sensor_adr[i]
            self.sensor_dim[name] = self._mj_model.sensor_dim[i]

    # -------------------- Sensor helper --------------------

    def _sensor(self, data, name: str) -> jax.Array:
        idx = self.sensor_adr[name]
        dim = self.sensor_dim[name]
        return data.sensordata[idx : idx + dim]

    # -------------------- Reset --------------------

    def reset(self, rng: jax.Array) -> mjx_env.State:
        _, subkey = jax.random.split(rng)

        # qpos0 → JAX
        qpos = jp.array(self._mj_model.qpos0)
        qvel = jp.zeros(self._mj_model.nv)

        # small perturbation on velocity
        qvel_noise = jax.random.normal(subkey, (6,)) * 0.02
        qvel = qvel.at[:6].add(qvel_noise)

        # renormalize quaternion
        qw, qx, qy, qz = qpos[3:7]
        norm = jp.sqrt(qw * qw + qx * qx + qy * qy + qz * qz)
        qpos = qpos.at[3:7].set(jp.array([qw, qx, qy, qz]) / norm)

        # ensure above floor (car body pos="0 0 .06" in XML)
        qpos = qpos.at[2].set(0.06)

        # Warp: make_data uses MjModel
        data = mjx_env.make_data(
            self._mj_model,
            qpos=qpos,
            qvel=qvel,
            impl=self._config.impl,
            naconmax=self._config.naconmax,
            njmax=self._config.njmax,
        )
        # forward uses mjx.Model
        data = mjx.forward(self._mjx_model, data)

        metrics: Dict[str, Any] = {}
        info: Dict[str, Any] = {}
        reward = jp.array(0.0)
        done = jp.array(0.0)
        obs = self._get_obs(data, info)

        # Set custom variables in info for handling 100hz gyro reads
        info["last_obs"] = obs
        info["obs_count"] = 0

        return mjx_env.State(data, obs, reward, done, metrics, info)

    # -------------------- Step --------------------

    def step(self, state: mjx_env.State, action: jax.Array) -> mjx_env.State:

        # increment observation counter to track when to read gyro (every 10 steps since sim_dt=0.002 and ctrl_dt=0.01)
        counter = state.info.get("obs_counter", 0) + 1
        counter = counter % 10  # 10 physics steps = 100 Hz

        # update info
        info = {"obs_counter": counter}

        # Normalize velociy so it is -1 to 1 for training stability (and to match action range)
        max_velocity = 20.0
        scaled_action = max_velocity * action
        scaled_action = jp.clip(scaled_action, -max_velocity, max_velocity)

        # physics step with mjx.Model
        next_data = mjx_env.step(
            self._mjx_model,
            state.data,
            scaled_action,
            self.n_substeps,
        )

        # Determine if robot has fallen over (based on pitch angle) or if there are NaNs in the state (indicating instability)
        # pitch angle
        q = next_data.qpos[3:7]
        qw, qx, qy, qz = q
        sinp = 2 * (qw * qy - qz * qx)
        theta = jp.arcsin(jp.clip(sinp, -1.0, 1.0))

        fallen = jp.abs(theta) > jp.deg2rad(30)
        nan_fail = jp.isnan(next_data.qpos).any() | jp.isnan(next_data.qvel).any()
        done = (fallen | nan_fail).astype(float)

        # SAFETY: stop motors if fallen
        scaled_action = jp.where(fallen, jp.zeros_like(scaled_action), scaled_action)

        reward = self._get_reward(next_data, scaled_action, state.info, state.metrics)
        obs = self._get_obs(next_data, state.info)

        reward = reward - 5.0 * fallen

        return mjx_env.State(next_data, obs, reward, done, state.metrics, state.info)

    # -------------------- Observation --------------------

    def _get_obs(self, data, info) -> jax.Array:
        # TODO: Add noise to observations

        counter = info.get("obs_counter", 0)
        # Only sample sensors every 10th physics step (100 Hz)
        if counter != 0:
            return info.get("last_obs")  # return previous observation

        # fetch real sensor values now at 100 hz
        gyro = self._sensor(data, "gyro")
        pitch_rate = gyro[1]  # pitch rate (y-axis)

        # Quaternion (orientation)
        q = data.qpos[3:7]
        qw, qx, qy, qz = q

        # Compute pitch angle from quaternion (same as in reward)
        sinp = 2 * (qw * qy - qz * qx)
        pitch_angle = jp.arcsin(jp.clip(sinp, -1.0, 1.0))  # pitch angle

        drift = data.qpos[0]
        x_dot = data.qvel[0]
        obs = jp.array([pitch_angle, pitch_rate, drift, x_dot])
        info["last_obs"] = obs

        # only return data robot actually has available
        return obs

    # -------------------- Reward --------------------

    def _get_reward(self, data, action, info, metrics) -> jax.Array:
        """
        Compute the reward signal for the robot control task.

        The reward encourages:
        1. Keeping the robot upright (pitch angle close to 0)
        2. Minimizing angular velocity (smooth motion)
        3. Staying near the origin (don't drift away)
        4. Using minimal control effort (energy efficiency)

        Returns:
            Total reward as the sum of four weighted reward components
        """
        # Extract pitch angle from quaternion (qpos[3:7] = [qw, qx, qy, qz])
        q = data.qpos[3:7]
        qw, qx, qy, qz = q
        # sin(pitch) = 2 * (qw*qy - qz*qx) using quaternion formula
        sinp = 2 * (qw * qy - qz * qx)
        theta = jp.arcsin(jp.clip(sinp, -1.0, 1.0))

        # Get angular velocity from IMU gyroscope sensor (y-axis = pitch rate)
        gyro = self._sensor(data, "gyro")
        theta_dot = gyro[1]

        # Extract x position and primary control input
        x = data.qpos[0]
        x_dot = data.qvel[0]  # <--- ADD THIS
        u = action[0]

        # Reward component 1: Penalize large pitch angles (encourages upright)
        # Gaussian penalty with scale 8.0, max reward = 1.0 at theta = 0
        r_theta = jp.exp(-8.0 * theta**2)

        # Reward component 2: Penalize excessive angular velocity (smooth motion)
        # Scaled to ~20% of theta reward, encourages damped motion
        r_thetadot = 0.4 * jp.exp(-0.1 * theta_dot**2)

        # Reward component 3: Penalize drift from center position (stay at origin)
        # Increased from 0.1 to 0.5 and scale from 0.5 to 2.0 for stronger centering
        r_x = 0.5 * jp.exp(-2.0 * x**2)

        # NEW: Penalize cart velocity (critical damping term)
        r_xdot = 0.6 * jp.exp(-1.0 * x_dot**2)  # <--- ADD THIS

        # Reward component 4: Penalize control effort (energy efficiency)
        # Negative reward for non-zero actions, encourages passive balancing
        r_u = -0.001 * (u**2)

        return r_theta + r_thetadot + r_x + r_xdot + r_u

    # -------------------- Properties --------------------

    @property
    def xml_path(self) -> str:
        return self._xml_path

    @property
    def action_size(self) -> int:
        return self._mjx_model.nu

    @property
    def position_size(self) -> int:
        return self._mjx_model.nq

    @property
    def positions(self) -> jax.Array:
        return self._mjx_model.qpos0

    @property
    def mj_model(self) -> mujoco.MjModel:
        return self._mj_model

    @property
    def mjx_model(self) -> mjx.Model:
        return self._mjx_model
