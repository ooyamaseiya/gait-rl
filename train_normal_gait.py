import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
import os
import imageio

class NormalGaitRewardWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        # Indices as per instructions:
        # 0: hull angle (hull_angle), 1: hull angular velocity (hull_ang_vel)
        # 4: leg1 hip angle (left), 9: leg2 hip angle (right)
        # 8: leg1 contact flag, 13: leg2 contact flag
        self.reset()

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.last_contact_leg = None  # Stores the last single-footed leg that made contact (1 or 2)
        self.prev_hip_diff_sign = 0   # -1, 0, 1
        self.leg1_contact_steps = 0
        self.leg2_contact_steps = 0
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        
        # Extracting values using provided indices
        hull_angle = obs[0]
        hull_ang_vel = obs[1]
        leg1_hip_angle = obs[4]
        leg2_hip_angle = obs[9]
        leg1_contact_flag = obs[8] > 0.5
        leg2_contact_flag = obs[13] > 0.5

        # 1. Posture penalty (hull angle and angular velocity)
        posture_penalty = -0.3 * np.abs(hull_angle) - 0.05 * np.abs(hull_ang_vel)

        # 2. Alternating leg contact bonus
        alternating_bonus = 0.0
        if (leg1_contact_flag and not leg2_contact_flag):  # Leg 1 only in contact
            if self.last_contact_leg == 2:
                alternating_bonus = 0.5
            self.last_contact_leg = 1
        elif (leg2_contact_flag and not leg1_contact_flag): # Leg 2 only in contact
            if self.last_contact_leg == 1:
                alternating_bonus = 0.5
            self.last_contact_leg = 2

        # 3. Hip crossover bonus (sign change of hip angle difference)
        crossover_bonus = 0.0
        hip_diff = leg1_hip_angle - leg2_hip_angle # obs[4] - obs[9]
        if np.abs(hip_diff) > 0.1:
            current_sign = int(np.sign(hip_diff))
            if self.prev_hip_diff_sign != 0 and current_sign != self.prev_hip_diff_sign:
                crossover_bonus = 0.5
            self.prev_hip_diff_sign = current_sign

        # 4. Drag penalty (continuous contact > 35 steps)
        drag_penalty = 0.0
        if leg1_contact_flag:
            self.leg1_contact_steps += 1
        else:
            self.leg1_contact_steps = 0

        if leg2_contact_flag:
            self.leg2_contact_steps += 1
        else:
            self.leg2_contact_steps = 0

        if self.leg1_contact_steps > 35 or self.leg2_contact_steps > 35:
            drag_penalty -= 0.1 # Applying penalty as a subtraction from total reward later via addition of negative value

        # Final custom reward calculation
        custom_reward = reward + posture_penalty + alternating_bonus + crossover_bonus + drag_penalty
        return obs, custom_reward, terminated, truncated, info

def record_video(model, step_label, video_dir):
    """Records a 1600-step video of the model's performance."""
    os.makedirs(video_dir, exist_ok=True)
    path = os.path.join(video_dir, f"step_{step_label}.mp4")

    # Create an environment for recording (non-vectorized to simplify frame extraction)
    env = gym.make("BipedalWalker-v3", render_mode="rgb_array")
    
    frames = []
    obs, info = env.reset()
    done = False
    truncated = False
    steps = 0

    # We use the trained model to act in a single episode
    while not (done or truncated) and steps < 1600:
        # Reshape obs for prediction as model expects batch dimension [N, dim]
        action, _states = model.predict(obs[None].astype(np.float32), deterministic=True)
        # action is shape (1, act_dim), so we take index 0 to get the control signal vector
        obs, reward, done, truncated, info = env.step(action[0])
        frames.append(env.render())
        steps += 1

    if frames:
        imageio.mimsave(path, frames, fps=30)
    else:
        print(f"Warning: No frames captured for {step_label}")
    
    env.close()

def main():
    # Directory preparation
    os.makedirs("./models", exist_ok=True)
    os.makedirs("./tb_logs", exist_ok=True)
    os.makedirs("./videos_normal", exist_ok=True)

    # 4-1 & 4-2: Training setup with SubprocVecEnv and NormalGaitRewardWrapper
    def make_env():
        return gym.make("BipedalWalker-v3")

    # Creating the vectorized environment wrapper around individual sub-environments
    train_env = SubprocVecEnv([lambda: NormalGaitRewardWrapper(gym.make("BipedalWalker-v3")) for _ in range(4)])

    # Initialize PPO with specific hyperparameters and tensorboard logging
    model = PPO(
        "MlpPolicy", 
        train_env, 
        verbose=1, 
        ent_coef=0.01, 
        tensorboard_log="./tb_logs/"
    )

    total_target_steps = 1_000_000
    chunk_size = 200_000
    num_chunks = total_target_steps // chunk_size

    try:
        for i in range(num_chunks):
            current_step = (i + 1) * chunk_size
            print(f"\n--- Training iteration {i+1}/{num_chunks} ({chunk_size} steps) ---")
            
            model.learn(total_timesteps=chunk_size, reset_num_timesteps=False, tb_log_name="PPO_NormalGait")
            
            # Recording video at every 200k milestone
            label = f"{current_step // 1000}k"
            print(f"Recording evaluation video for {label}...")
            record_video(model, label.replace(' ', '_'), "./videos_normal")

        # Final model saving
        final_save_path = "./models/ppo_bipedal_normal_1000k"
        model.save(final_save_path)
        print(f"\nTraining complete. Model saved at: {final_save_path}")

    except Exception as e:
        print(f"An error occurred during training: {e}")
        raise e
    finally:
        train_env.close()

if __name__ == "__main__":
    main()
