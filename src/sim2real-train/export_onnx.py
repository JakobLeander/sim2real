"""
Correct ONNX exporter for Brax PPO using JAX tracing.

Key idea:
    NO PyTorch, NO manual layer reconstruction.
    We export the REAL JAX policy directly.
"""

import numpy as np
import pickle
import jax
import jax.numpy as jnp
from etils import epath
import onnxruntime as ort
import jax2onnx

import robot
from playground import wrapper


# =========================================================
# Load trained params
# =========================================================
def load_params():
    policies_dir = epath.Path("policies")
    with open(policies_dir / "robot_policy.pkl", "rb") as f:
        return pickle.load(f)


# =========================================================
# Extract inference function (IMPORTANT)
# =========================================================
def get_inference_fn(env, params):
    from brax.training.agents.ppo import networks as ppo_networks

    networks = ppo_networks.make_ppo_networks(
        observation_size=env.observation_size,
        action_size=env.action_size,
    )
    make_inference_fn = ppo_networks.make_inference_fn(networks)
    return make_inference_fn(params, deterministic=True)


# =========================================================
# JAX policy wrapper (THIS is what we export)
# =========================================================
def build_jax_policy(inference_fn):
    def policy(obs):
        rng = jax.random.PRNGKey(0)
        action, _ = inference_fn(obs, rng)
        return action

    return policy


# =========================================================
# Convert JAX → ONNX directly
# =========================================================
def export_to_onnx(jax_policy, obs_shape, onnx_path):
    from jax import ShapeDtypeStruct

    print("\n🔄 Converting JAX → ONNX directly with jax2onnx...")

    jax2onnx.to_onnx(
        jax_policy,
        [ShapeDtypeStruct((1, obs_shape), jnp.float32)],
        output_path=str(onnx_path),
        opset=13,
        input_names=["obs"],
        output_names=["action"],
        return_mode="file",
    )

    print(f"✅ ONNX saved to {onnx_path}")

    # Optimize the ONNX model for faster inference
    optimize_onnx_model(onnx_path)


# =========================================================
# Optimize ONNX model for inference
# =========================================================
def optimize_onnx_model(onnx_path):
    import onnx
    from onnxoptimizer import optimize

    print("\n⚡ Optimizing ONNX model for faster inference...")

    # Load the model
    model = onnx.load(str(onnx_path))

    # Optimize the model
    optimized_model = optimize(model)

    # Save the optimized model back
    onnx.save(optimized_model, str(onnx_path))

    print(f"✅ Optimized ONNX saved to {onnx_path}")


# =========================================================
# Validation: JAX vs ONNX
# =========================================================
def validate(onnx_path, jax_policy, obs_shape):
    print("\n🧪 Running validation...")

    # Optimize session for inference
    sess_options = ort.SessionOptions()
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    sess_options.execution_mode = (
        ort.ExecutionMode.ORT_SEQUENTIAL
    )  # Suitable for Raspberry Pi

    session = ort.InferenceSession(str(onnx_path), sess_options=sess_options)
    input_name = session.get_inputs()[0].name

    errors = []

    for _ in range(50):
        obs = np.random.randn(1, obs_shape).astype(np.float32)

        rng = jax.random.PRNGKey(0)
        jax_out = jax_policy(obs)[0]

        onnx_out = session.run(None, {input_name: obs})[0]

        err = np.abs(jax_out - onnx_out).mean()
        errors.append(err)

    errors = np.array(errors)

    print("\n📊 RESULTS")
    print("Mean error:", errors.mean())
    print("Max error :", errors.max())

    if errors.mean() < 1e-4:
        print("✅ PASS: ONNX matches JAX")
    else:
        print("⚠️ FAIL: mismatch")


# =========================================================
# MAIN
# =========================================================
def main():
    print("🚀 JAX → ONNX Export (Correct Pipeline)")

    env_cfg = robot.default_config()
    env = robot.Robot(env_cfg)

    obs_size = env.observation_size

    print(f"Obs size: {obs_size}")

    # -----------------------------
    # Load trained params
    # -----------------------------
    params = load_params()

    # -----------------------------
    # Build inference function
    # -----------------------------
    print("🧠 Rebuilding inference function from saved policy params...")
    inference_fn = get_inference_fn(env, params)
    jax_policy = build_jax_policy(inference_fn)

    # -----------------------------
    # Export ONNX
    # -----------------------------
    onnx_path = epath.Path("policies/robot_policy.onnx")

    export_to_onnx(jax_policy, obs_size, onnx_path)

    # -----------------------------
    # Validate
    # -----------------------------
    validate(onnx_path, jax_policy, obs_size)


if __name__ == "__main__":
    main()
