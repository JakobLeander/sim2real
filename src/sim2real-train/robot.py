from typing import Any, Dict, Optional, Union

import jax
import jax.numpy as jp
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
        ctrl_dt=0.02,
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

        # ensure above floor (car body pos="0 0 .4" in XML)
        qpos = qpos.at[2].set(0.4)

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

        return mjx_env.State(data, obs, reward, done, metrics, info)

    # -------------------- Step --------------------

    def step(self, state: mjx_env.State, action: jax.Array) -> mjx_env.State:
        # physics step with mjx.Model
        next_data = mjx_env.step(
            self._mjx_model,
            state.data,
            action,
            self.n_substeps,
        )

        reward = self._get_reward(next_data, action, state.info, state.metrics)
        obs = self._get_obs(next_data, state.info)

        # pitch angle
        q = next_data.qpos[3:7]
        qw, qx, qy, qz = q
        sinp = 2 * (qw * qy - qz * qx)
        theta = jp.arcsin(jp.clip(sinp, -1.0, 1.0))

        fallen = jp.abs(theta) > jp.deg2rad(30)
        nan_fail = jp.isnan(next_data.qpos).any() | jp.isnan(next_data.qvel).any()
        done = (fallen | nan_fail).astype(float)

        reward = reward - 5.0 * fallen

        return mjx_env.State(next_data, obs, reward, done, state.metrics, state.info)

    # -------------------- Observation --------------------

    def _get_obs(self, data, info) -> jax.Array:
        qpos = data.qpos[:7]
        qvel = data.qvel[:6]

        gyro = self._sensor(data, "gyro")
        accel = self._sensor(data, "accelerometer")
        linvel = self._sensor(data, "local_linvel")

        return jp.concatenate([qpos, qvel, gyro, accel, linvel])

    # -------------------- Reward --------------------

    def _get_reward(self, data, action, info, metrics) -> jax.Array:
        q = data.qpos[3:7]
        qw, qx, qy, qz = q
        sinp = 2 * (qw * qy - qz * qx)
        theta = jp.arcsin(jp.clip(sinp, -1.0, 1.0))

        gyro = self._sensor(data, "gyro")
        theta_dot = gyro[1]

        x = data.qpos[0]
        u = action[0]

        r_theta = jp.exp(-8.0 * theta**2)
        r_thetadot = 0.2 * jp.exp(-0.1 * theta_dot**2)
        r_x = 0.1 * jp.exp(-0.5 * x**2)
        r_u = -0.001 * (u**2)

        return r_theta + r_thetadot + r_x + r_u

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
