from email import parser

import onnxruntime

class OnnxInfer:
    ONNX_POLICY_PATH = "policies/robot_policy.onnx"
    OBS_SIZE = 4
    AWD=True
    
    def __init__(self):
        self.ort_session = onnxruntime.InferenceSession(
            self.ONNX_POLICY_PATH, providers=["CPUExecutionProvider"]
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
    import argparse
    import numpy as np
    import time

    oi = OnnxInfer()
    times = []
    for i in range(1000):
        inputs = np.random.uniform(size=OnnxInfer.OBS_SIZE).astype(np.float32)
        # inputs = np.arange(obs_size).astype(np.float32)
        # print(inputs)
        start = time.time()
        action = oi.infer(inputs)
        
        print("type:", type(action))
        print(float(action[0]))
        times.append(time.time() - start)

    print("Average time: ", sum(times) / len(times))
    print("Average fps: ", 1 / (sum(times) / len(times)))
