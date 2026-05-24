"""
Run the trained policy in a simulator
"""

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
        onnx_policy_path = "policies/robot_policy.onnx"
        self.model = mujoco.MjModel.from_xml_string(robot_xml_path.read_text())

        self.sim_dt = 0.001  # physics runs at 1000 hz
        self.model.opt.timestep = self.sim_dt
        self.data = mujoco.MjData(self.model)
        mujoco.mj_step(self.model, self.data)

        self.policy = OnnxInfer(onnx_policy_path, awd=True)

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
            max_velocity = 20.0
            MAX_SPEED_CHANGE = 2.0  # must match robot.py training constraint
            last_action = np.zeros(self.model.nu)

            # print observation size
            print(f"Observation size: {len(self.get_obs(self.data, 0.0))}")

            while v.is_running():
                step_start = time.time()

                mujoco.mj_step(self.model, self.data)

                counter += 1

                # our action loop runs at 1000 hz, so we only infer every 10 steps to simular physical sensor readings at 100 hz (gyro is the limiting factor since it runs at 100 hz in real life)
                if counter % 10 == 0:
                    obs = self.get_obs(self.data, last_action[0])
                    action = self.policy.infer(obs)

                    # Unnormalize
                    action = max_velocity * action

                    # Rate-limit: match the training constraint in robot.py
                    delta = np.clip(
                        action - last_action, -MAX_SPEED_CHANGE, MAX_SPEED_CHANGE
                    )
                    action = np.clip(last_action + delta, -max_velocity, max_velocity)
                    last_action = action.copy()

                    if self.nudge:
                        nudge_counter += 1
                        action += np.random.uniform(2.0, 2.0, size=action.shape)
                        action = np.clip(action, -max_velocity, max_velocity)
                        last_action = action.copy()
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
