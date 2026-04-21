"""
Run the trained policy in a simulator
"""

from typing import Any, Dict, Optional, Union
import mujoco
import pickle
import numpy as np
import mujoco
import mujoco.viewer
import time
from etils import epath


class RobotRunner:
    def __init__(self):
        robot_xml_path = epath.Path(__file__).parent / "xmls/robot.xml"
        self.model = mujoco.MjModel.from_xml_string(robot_xml_path.read_text())

        self.sim_dt = 0.002
        self.decimation = 10
        self.model.opt.timestep = self.sim_dt
        self.data = mujoco.MjData(self.model)
        mujoco.mj_step(self.model, self.data)

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
        print(f"key: {keycode}")

    def run(self):
        with mujoco.viewer.launch_passive(
            self.model,
            self.data,
            show_left_ui=True,
            show_right_ui=True,
            key_callback=self.key_callback,
        ) as viewer:
            counter = 0
            while True:
                step_start = time.time()

                mujoco.mj_step(self.model, self.data)

                counter += 1

                viewer.sync()

                time_until_next_step = self.model.opt.timestep - (
                    time.time() - step_start
                )
                if time_until_next_step > 0:
                    time.sleep(time_until_next_step)


if __name__ == "__main__":

    runner = RobotRunner()
    runner.run()
