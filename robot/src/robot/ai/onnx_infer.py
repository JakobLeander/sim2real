from pathlib import Path
import numpy as np
import time

import onnxruntime


class OnnxInfer:
    AWD = True

    def __init__(self, onnx_policy_path: str):
        self.ort_session = onnxruntime.InferenceSession(
            onnx_policy_path, providers=["CPUExecutionProvider"]
        )
        self.input_name = self.ort_session.get_inputs()[0].name

    def infer(self, inputs):
        if self.AWD:
            outputs = self.ort_session.run(None, {self.input_name: [inputs]})
            return outputs[0][0]
        else:
            outputs = self.ort_session.run(
                None, {self.input_name: inputs.astype("float32")}
            )
            return outputs[0]


if __name__ == "__main__":
    OBS_SIZE = 3
    onnx_policy_path = Path(__file__).parents[4] / "policies" / "robot_policy.onnx"

    oi = OnnxInfer(str(onnx_policy_path))
    times = []
    for i in range(1000):
        inputs = np.random.uniform(size=OBS_SIZE).astype(np.float32)
        # inputs = np.arange(obs_size).astype(np.float32)
        # print(inputs)
        start = time.time()
        action = oi.infer(inputs)

        print(f"Action: {float(action[0])}")
        times.append(time.time() - start)

    print("Average time: ", sum(times) / len(times))
    print("Average fps: ", 1 / (sum(times) / len(times)))
