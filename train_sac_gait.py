import os

import gymnasium as gym
import imageio
from stable_baselines3 import SAC


def record_video(env_id, model, filename, steps=1600):
    eval_env = gym.make(env_id, render_mode="rgb_array")
    obs, _ = eval_env.reset()
    frames = []

    for _ in range(steps):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = eval_env.step(action)
        frame = eval_env.render()
        if frame is not None:
            frames.append(frame)
        if terminated or truncated:
            break

    eval_env.close()
    if frames:
        imageio.mimsave(filename, frames, fps=30)
        return True
    return False


if __name__ == "__main__":
    os.makedirs("./videos_sac", exist_ok=True)
    os.makedirs("./models", exist_ok=True)
    os.makedirs("./tb_logs", exist_ok=True)

    ENV_ID = "BipedalWalker-v3"
    INCREMENTAL_STEPS = 100_000
    NUM_CYCLES = 5  # 合計 500,000 ステップ
    MODEL_PATH = "./models/sac_bipedal_500k"

    # SACはリプレイバッファを使うため、単一環境で回すのが基本
    train_env = gym.make(ENV_ID)

    model = SAC(
        "MlpPolicy",
        train_env,
        verbose=1,
        tensorboard_log="./tb_logs/",
    )

    print("Starting SAC training...")
    for i in range(NUM_CYCLES):
        current_total = (i + 1) * INCREMENTAL_STEPS
        print(
            f"\n--- Cycle {i + 1}/{NUM_CYCLES}: Training up to {current_total // 1000}k steps ---")
        model.learn(
            total_timesteps=INCREMENTAL_STEPS,
            reset_num_timesteps=False,
            tb_log_name="SAC_Bipedal",
        )

        video_file = f"./videos_sac/step_{current_total // 1000}k.mp4"
        print(f"Recording video to {video_file}")
        record_video(ENV_ID, model, video_file)

    model.save(MODEL_PATH)
    print(f"\nTraining complete. Model saved to {MODEL_PATH}.zip")
    train_env.close()
