"""
Run the trained policy in a simulator
"""

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

        self.sim_dt = 0.02
        self.decimation = 10
        self.model.opt.timestep = self.sim_dt
        self.data = mujoco.MjData(self.model)
        mujoco.mj_step(self.model, self.data)

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
