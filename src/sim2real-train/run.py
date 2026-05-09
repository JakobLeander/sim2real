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
from helper.onnx_infer import OnnxInfer


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

    def get_obs(self, data):
        qpos = data.qpos[:7]
        qvel = data.qvel[:6]

        gyro = self.sensor(data, "gyro")
        accel = self.sensor(data, "accelerometer")
        linvel = self.sensor(data, "local_linvel")

        return np.concatenate([qpos, qvel, gyro, accel, linvel])

    def run(self):
        with viewer.launch_passive(
            self.model, self.data, key_callback=self.key_callback
        ) as v:

            counter = 0
            nudge_counter = 0
            while v.is_running():
                step_start = time.time()

                mujoco.mj_step(self.model, self.data)

                counter += 1

                # our action loop runs at 100 hz, so we only infer every 10 steps
                if counter % 10 == 0:
                    obs = self.get_obs(self.data)
                    action = self.policy.infer(obs)

                    # since we normalized actions during training, we need to unnormalize them here
                    max_velocity = 20.0
                    action = max_velocity * action

                    if self.nudge:
                        nudge_counter += 1
                        action += np.random.uniform(10.0, 10.0, size=action.shape)
                        if nudge_counter > 10:
                            self.nudge = False
                            nudge_counter = 0

                    self.data.ctrl[:] = action
                    counter = 0

                v.sync()

                time_until_next_step = self.model.opt.timestep - (
                    time.time() - step_start
                )
                if time_until_next_step > 0:
                    time.sleep(time_until_next_step)


if __name__ == "__main__":

    runner = RobotRunner()
    runner.run()
