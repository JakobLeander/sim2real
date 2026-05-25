"""
Run the trained policy in a simulator
"""

import sys
from typing import Any, Dict, Optional, Union
import mujoco
import pickle
import numpy as np
import mujoco
import mujoco.viewer as viewer
import time
from etils import epath
from warp import jax
from training.helper.onnx_infer import OnnxInfer

# Add repo root to path so shared/ package is importable
_REPO_ROOT = str(epath.Path(__file__).parents[3])
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from shared.constants import RobotSpec
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    filename="run_simulation.log",
    filemode="w",
)


class RobotRunner:
    def __init__(self):
        robot_xml_path = epath.Path(__file__).parent / "xmls/robot.xml"
        onnx_path = epath.Path(__file__).parents[3] / "policies" / "robot_policy.onnx"
        self.model = mujoco.MjModel.from_xml_string(robot_xml_path.read_text())

        self.sim_dt = 0.001  # physics runs at 1000 hz
        self.ctrl_dt = RobotSpec.CONTROL_DT

        self.model.opt.timestep = self.sim_dt
        self.data = mujoco.MjData(self.model)
        mujoco.mj_step(self.model, self.data)

        self.policy = OnnxInfer(onnx_path, awd=True)

        self.nudge = False

        # Build sensor lookup tables (version‑safe)
        self.sensor_adr: Dict[str, int] = {}
        self.sensor_dim: Dict[str, int] = {}

        names_buf = self.model.names

        if hasattr(self.model, "sensor_nameadr"):
            nameadr = self.model.sensor_nameadr
        elif hasattr(self.model, "name_sensoradr"):
            nameadr = self.model.name_sensoradr
        else:
            raise RuntimeError("MuJoCo model has no sensor name address field.")

        for i in range(self.model.nsensor):
            adr = nameadr[i]
            name_bytes = names_buf[adr:].split(b"\x00", 1)[0]
            name = name_bytes.decode("utf-8")

            self.sensor_adr[name] = self.model.sensor_adr[i]
            self.sensor_dim[name] = self.model.sensor_dim[i]

    # -------------------- Sensor helper --------------------
    def sensor(self, data, name: str):
        idx = self.sensor_adr[name]
        dim = self.sensor_dim[name]
        return data.sensordata[idx : idx + dim]

    def key_callback(self, keycode):
        self.nudge = True
        print("Nudge!")

    def get_obs(self, data, last_u: float) -> np.ndarray:
        # Quaternion (orientation)
        q = data.qpos[3:7]
        qw, qx, qy, qz = q

        # Compute pitch angle from quaternion (same as in reward)
        sinp = 2 * (qw * qy - qz * qx)
        pitch_angle = np.arcsin(np.clip(sinp, -1.0, 1.0))  # pitch angle

        gyro = self.sensor(data, "gyro")
        pitch_rate = gyro[1]  # pitch rate (y-axis)

        # Normalise last_u to [-1, 1] — must match robot.py training
        last_u_normalised = last_u / 20.0

        return np.array([pitch_angle, pitch_rate, last_u_normalised], dtype=np.float32)

    def write_log(self, message: str):
        logging.info(message)

    def run(self):
        with viewer.launch_passive(
            self.model, self.data, key_callback=self.key_callback
        ) as v:

            counter = 0
            nudge_counter = 0
            MAX_VELOCITY = RobotSpec.MAX_VELOCITY
            MAX_DELTA_VEL = RobotSpec.MAX_DELTA_VEL
            CTRL_DT_COUNT = round(self.ctrl_dt / self.sim_dt)

            last_action = np.zeros(self.model.nu)

            logging.info("=" * 60)
            logging.info("RobotRunner starting")
            logging.info("=" * 60)
            logging.info(
                f"  sim_dt:        {self.sim_dt:.4f} s  ({1/self.sim_dt:.0f} Hz)"
            )
            logging.info(
                f"  ctrl_dt:       {self.ctrl_dt:.4f} s  ({1/self.ctrl_dt:.0f} Hz)"
            )
            logging.info(f"  ctrl_dt_count: {CTRL_DT_COUNT} sim steps per control step")
            logging.info(f"  max_velocity:  {MAX_VELOCITY} rad/s")
            logging.info(f"  max_delta_vel: {MAX_DELTA_VEL} rad/s per control step")
            logging.info(f"  nu (actuators):{self.model.nu}")
            logging.info(f"  nq:            {self.model.nq}")
            logging.info(f"  nv:            {self.model.nv}")
            logging.info(f"  obs_size:      {len(self.get_obs(self.data, 0.0))}")
            logging.info("=" * 60)

            while v.is_running():
                step_start = time.time()

                mujoco.mj_step(self.model, self.data)

                counter += 1

                # Insure control loop running at same rate as physical robot
                if counter % CTRL_DT_COUNT == 0:
                    obs = self.get_obs(self.data, last_action[0])
                    action = self.policy.infer(obs)

                    # Unnormalize
                    action = MAX_VELOCITY * action

                    # Rate-limit: match the training constraint in robot.py
                    delta = np.clip(action - last_action, -MAX_DELTA_VEL, MAX_DELTA_VEL)
                    action = np.clip(last_action + delta, -MAX_VELOCITY, MAX_VELOCITY)
                    last_action = action.copy()

                    if self.nudge:
                        action += np.random.uniform(2.0, 2.0, size=action.shape)
                        action = np.clip(action, -MAX_VELOCITY, MAX_VELOCITY)
                        last_action = action.copy()
                        logging.info(
                            f"NUDGE [{nudge_counter}/10]: Motor Speed: {action[0]:.4f} rad/s"
                        )
                        nudge_counter += 1

                        if nudge_counter > 10:
                            self.nudge = False
                            nudge_counter = 0

                    self.data.ctrl[:] = action
                    counter = 0
                    self.write_log(
                        f"Pitch Angle: {obs[0]:.4f} rad, Pitch Velocity: {obs[1]:.4f} rad/s, Motor Speed: {action[0]:.4f} rad/s"
                    )
                    v.sync()

                time_until_next_step = self.model.opt.timestep - (
                    time.time() - step_start
                )
                if time_until_next_step > 0:
                    time.sleep(time_until_next_step)


if __name__ == "__main__":

    runner = RobotRunner()
    runner.run()
