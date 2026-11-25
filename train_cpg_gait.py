import os

import gymnasium as gym
import imageio
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv


class CPGGaitWrapper(gym.Wrapper):
    """
    CPG（歩行リズム位相）を観測に追加し、
    左右の股関節が交互に振り出されるよう連続的な報酬を与えるラッパー
    """

    def __init__(self, env, freq=0.025, amplitude=0.7):
        super().__init__(env)
        # freq: 歩行のテンポ（1ステップあたりの位相の進み具合。0.025で約40ステップ=1周期）
        # amplitude: 左右の股関節角度差の目標振幅（大きいほど大股になる）
        self.freq = freq
        self.amplitude = amplitude
        self.step_count = 0

        # 観測空間を 24次元 → 26次元（元の観測 + sin(phase), cos(phase)）に拡張
        orig_low = self.observation_space.low
        orig_high = self.observation_space.high
        low = np.concatenate(
            [orig_low, np.array([-1.0, -1.0], dtype=np.float32)])
        high = np.concatenate(
            [orig_high, np.array([1.0, 1.0], dtype=np.float32)])
        self.observation_space = gym.spaces.Box(
            low=low, high=high, dtype=np.float32)

    def _get_phase_features(self):
        phase = 2.0 * np.pi * self.freq * self.step_count
        return np.array([np.sin(phase), np.cos(phase)], dtype=np.float32)

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.step_count = 0
        aug_obs = np.concatenate([obs, self._get_phase_features()])
        return aug_obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.step_count += 1

        # 現在の歩行リズム（sin, cos）を計算
        phase_features = self._get_phase_features()
        sin_phase = phase_features[0]

        # 必要なセンサー値を取得
        torso_angle = obs[0]     # 体幹の傾き
        torso_ang_vel = obs[1]   # 体幹の角速度
        vel_x = obs[2]           # 前進速度
        hip1_angle = obs[4]      # 脚1（左）の股関節角度
        hip2_angle = obs[9]      # 脚2（右）の股関節角度

        # 1. 体幹の直立保持ペナルティ
        posture_penalty = -0.3 * abs(torso_angle) - 0.05 * abs(torso_ang_vel)

        # 2. CPGによる左右交互の股関節振り出し報酬（Dense Reward）
        # 左右の股関節角度差が目標の正弦波（±amplitude）に近いほど高得点（最大 +0.4 / step）
        target_hip_diff = self.amplitude * sin_phase
        actual_hip_diff = hip1_angle - hip2_angle
        hip_error = abs(actual_hip_diff - target_hip_diff)
        gait_rhythm_bonus = 0.4 * np.exp(-3.0 * (hip_error ** 2))

        # 止まったまま脚だけバタバタするズル（報酬ハッキング）を防ぐため、
        # 前進しているときだけリズム報酬を有効にする
        if vel_x < 0.05:
            gait_rhythm_bonus *= 0.1

        # 最終報酬
        custom_reward = reward + posture_penalty + gait_rhythm_bonus

        # 拡張した観測データ（26次元）を返す
        aug_obs = np.concatenate([obs, phase_features])
        return aug_obs, custom_reward, terminated, truncated, info


def record_video(env_id, model, filename, steps=1600):
    """評価用の動画を録画する（モデル入力が26次元のため同じラッパーを適用）"""
    eval_env = CPGGaitWrapper(gym.make(env_id, render_mode="rgb_array"))
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
    os.makedirs("./videos_cpg", exist_ok=True)
    os.makedirs("./models", exist_ok=True)
    os.makedirs("./tb_logs", exist_ok=True)

    ENV_ID = "BipedalWalker-v3"
    NUM_ENVS = 4
    INCREMENTAL_STEPS = 200_000
    NUM_CYCLES = 5  # 合計 1,000,000 ステップ
    MODEL_PATH = "./models/ppo_bipedal_cpg_1000k"

    def make_env():
        return CPGGaitWrapper(gym.make(ENV_ID))

    print("Initializing vectorized environment...")
    train_env = SubprocVecEnv([make_env for _ in range(NUM_ENVS)])

    model = PPO(
        "MlpPolicy",
        train_env,
        verbose=1,
        learning_rate=3e-4,
        ent_coef=0.01,
        tensorboard_log="./tb_logs/",
    )

    print("Starting CPG-guided training...")
    for i in range(NUM_CYCLES):
        current_total = (i + 1) * INCREMENTAL_STEPS
        print(
            f"\n--- Cycle {i + 1}/{NUM_CYCLES}: Training up to {current_total // 1000}k steps ---")
        model.learn(
            total_timesteps=INCREMENTAL_STEPS,
            reset_num_timesteps=False,
            tb_log_name="PPO_CPGGait",
        )

        video_file = f"./videos_cpg/step_{current_total // 1000}k.mp4"
        print(f"Recording video to {video_file}")
        record_video(ENV_ID, model, video_file)

    model.save(MODEL_PATH)
    print(f"\nTraining complete. Model saved to {MODEL_PATH}.zip")
    train_env.close()
