import gymnasium as gym
import numpy as np
import os
import imageio
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv

class AlternatingGaitRewardWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.reset_internal()

    def reset_internal(self):
        # self.prev_lead_leg: 1=left leg (obs[4] > obs[9]), -1=right leg (obs[4] < obs[9]), 0=unknown/uninitialized
        self.prev_lead_leg = 0
        self.same_lead_steps = 0
        self.waiting_for_contact_leg = 0  # 1 or 2, 0 if none
        self.leg1_continuous_contact = 0
        self.leg2_continuous_contact = 0

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.reset_internal()
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)

        # Indices from instructions:
        # 0: torso angle
        # 1: torso angular velocity
        # 2: vel_x (forward speed)
        # 4: hip 1 (left) angle
        # 9: hip 2 (right) angle
        # 8: leg 1 contact flag (>0.5 for grounded)
        # 13: leg 2 contact flag (>0.5 for grounded)

        torso_angle = obs[0]
        torso_ang_vel = obs[1]
        vel_x = obs[2]
        hip1_angle = obs[4]
        hip2_angle = obs[9]
        leg1_contact = obs[8] > 0.5
        leg2_contact = obs[13] > 0.5

        # 1. Posture penalty (as specified)
        posture_penalty = -0.3 * abs(torso_angle) - 0.05 * abs(torso_ang_vel)

        # 2. Crossover and stagnation logic
        hip_diff = hip1_angle - hip2_angle
        crossover_bonus = 0.0
        stagnation_penalty = 0.0
        current_lead = 0
        if abs(hip_diff) > 0.15:
            # Instruction says current_lead = 1 if hip_diff > 0 else -1
            current_lead = 1 if hip_diff > 0 else -1

            if self.prev_lead_leg != 0 and current_lead != self.prev_lead_leg:
                # Crossover (Leg swap) detected!
                if vel_x > 0.05:
                    crossover_bonus = 1.0
                self.waiting_for_contact_leg = 1 if current_lead == 1 else 2
                self.same_lead_steps = 0
                self.prev_lead_leg = current_lead
            elif self.prev_lead_leg == 0:
                # First time a lead leg is established (start of gait cycle)
                self.prev_lead_leg = current_lead
                self.same_lead_steps = 1 # Initializing count for the first state
            else:
                # Same lead leg continues to be in front
                if abs(hip_diff) > 0.15 and current_lead == self.prev_lead_leg:
                    self.same_lead_steps += 1

        if self.same_lead_steps > 50:
            stagnation_penalty = -0.15

        # 3. "振り出した脚の接地" ボーナス (step contact bonus for the newly lead leg)
        step_contact_bonus = 0.0
        if self.waiting_for_contact_leg == 1 and leg1_contact:
            step_contact_bonus = 2.0
            self.waiting_for_contact_leg = 0
        elif self.waiting_for_contact_leg == 2 and leg2_contact:
            step_contact_bonus = 2.0
            self.waiting_for_contact_leg = 0

        # 4. Drag penalty (continuous contact > 30)
        drag_penalty = 0.0
        if leg1_contact:
            self.leg1_continuous_contact += 1
        else:
            self.leg1_continuous_contact = 0

        if leg2_contact:
            self.leg2_continuous_contact += 1
        else:
            self.leg2_continuous_contact = 0

        if self.leg1_continuous_contact > 30 or self.leg2_continuous_contact > 30:
            drag_penalty = -0.1

        # Final combined reward
        custom_reward = reward + posture_penalty + crossover_bonus + step_contact_bonus + stagnation_penalty + drag_penalty
        return obs, custom_reward, terminated, truncated, info


def record_video(env_id, model, filename, steps=1600):
    # Record using original environment for clean visualization (no reward wrapper)
    eval_env = gym.make(env_id, render_mode="rgb_array")
    obs, _ = eval_env.reset()
    frames = []

    for _ in range(steps):
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = eval_env.step(action)
        frame = eval_env.render()
        if frame is not None:
            frames.append(frame)
        if terminated or truncated:
            break

    eval_env.close()
    if len(frames) > 0:
        imageio.mimsave(filename, frames, fps=30)
        return True
    return False


if __name__ == "__main__":
    # Ensure directories exist for outputs
    os.makedirs("./videos_alternating", exist_ok=True)
    os.makedirs("./models", exist_ok=True)
    os.makedirs("./tb_logs", exist_ok=True)

    ENV_ID = "BipedalWalker-v3"
    NUM_ENVS = 4
    TOTAL_STEPS = 1_000_000
    INCREMENTAL_STEPS = 200_000
    MODEL_PATH = "./models/ppo_bipedal_alternating_1000k"

    def make_env():
        # Using the wrapper for training to implement reward shaping logic
        return AlternatingGaitRewardWrapper(gym.make(ENV_ID))

    print("Initializing vectorized environment...")
    train_env = SubprocVecEnv([make_env for _ in range(NUM_ENVS)])

    # PPO Configuration as specified in instructions
    model = PPO(
        "MlpPolicy", 
        train_env, 
        verbose=1, 
        ent_coef=0.015, 
        tensorboard_log="./tb_logs/"
    )

    print("Starting training loop...")
    for i in range(5):
        current_total = (i + 1) * INCREMENTAL_STEPS
        print(f"\n--- Cycle {i+1}/5: Training up to {current_total}k steps ---")
        model.learn(
            total_timesteps=INCREMENTAL_STEPS, 
            reset_num_timesteps=False, 
            tb_log_name="PPO_AlternatingGait"
        )
        
        # Record evaluation video every 200k steps
        video_file = f"./videos_alternating/step_{current_total // 1000}k.mp4"
        print(f"Recording video to {video_file}")
        success = record_video(ENV_ID, model, video_file)
        if not success:
            print("Warning: Video recording failed (episode ended too early or no frames).")

    # Final save of the trained agent
    model.save(MODEL_PATH)
    print(f"\nTraining complete.")
    print(f"Model saved to {MODEL_PATH}.zip")

    train_env.close()
