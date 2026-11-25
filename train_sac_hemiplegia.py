import gymnasium as gym
import numpy as np
from stable_baselines3 import SAC
import imageio
import os

class HemiplegiaWrapper(gym.Wrapper):
    def __init__(self, env, strength_ratio=0.3):
        super().__init__(env)
        self.strength_ratio = strength_ratio

    def step(self, action):
        # Action space: [Leg1_Hip, Leg1_Knee, Leg2_Hip, Leg2_Knee] (4 dims)
        modified_action = np.array(action, dtype=np.float32, copy=True)
        
        # Apply muscle strength limitation to Leg 2 (indices 2 and 3)
        modified_action[2] *= self.strength_ratio
        modified_action[3] *= self.strength_ratio
        
        obs, reward, terminated, truncated, info = self.env.step(modified_action)
        
        # Reward bonus for survival (not being in a terminal state/falling)
        if not terminated:
            reward += 0.05
            
        return obs, reward, terminated, truncated, info

def record_video(env_id, model, filename, strength_ratio=0.3, steps=1600):
    print(f"Recording video to {filename}...")
    # Important: Use the wrapped environment for evaluation too!
    eval_env = gym.make(env_id, render_mode="rgb_array")
    eval_env = HemiplegiaWrapper(eval_env, strength_ratio=strength_ratio)
    
    obs, _ = eval_env.reset()
    frames = []
    
    for _ in range(steps):
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = eval_env.step(action)
        frames.append(eval_env.render())
        if terminated or truncated:
            break
            
    imageio.mimsave(filename, frames, fps=30)
    eval_env.close()
    print(f"Video saved to {filename}")

if __name__ == "__main__":
    # Paths and setup
    MODEL_PATH = "./models/sac_bipedal_500k.zip"
    VIDEO_DIR = "./videos_hemiplegia"
    LOGS_DIR = "./tb_logs/" # SB3 will create subdirs inside this
    FINAL_MODEL_NAME = "./models/sac_bipedal_hemiplegia_300k"

    # Check if base model exists
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Base trained model NOT found at {MODEL_PATH}. Please check the path.")

    for d in [VIDEO_DIR, "./models", LOGS_DIR]:
        os.makedirs(d, exist_ok=True)

    # Create training environment (wrapped with 30% strength for Leg 2)
    train_env = gym.make("BipedalWalker-v3")
    train_env = HemiplegiaWrapper(train_env, strength_ratio=0.3)

    print(f"Loading pre-trained model: {MODEL_PATH}")
    # Note: SB3 load usually takes the filename without extension or with it works fine for .zip files in most versions.
    model = SAC.load(
        MODEL_PATH, 
        env=train_env,
        tensorboard_log=LOGS_DIR,
    )

    # 1. Recording at step 0 (Before fine-tuning in the new environment context?)
    print("Recording baseline (Step 0k) - Normal brain in hemiplegic environment...")
    record_video("BipedalWalker-v3", model, os.path.join(VIDEO_DIR, "step_000k.mp4"), strength_ratio=0.3)

    # 2. Fine-tuning Loop
    total_steps = 300_000
    chunk_size = 100_000
    
    for i in range(3):
        current_step_in_loop = (i + 1) * chunk_size
        print(f"Starting training: Chunk {i+1}/3 ({current_step_in_loop}k steps total)...")
        
        model.learn(total_timesteps=chunk_size, reset_num_timesteps=False, tb_log_name="SAC_Hemiplegia")
        
        # Record video after each chunk
        video_filename = os.path.join(VIDEO_DIR, f"step_{current_step_in_loop//1000}k.mp4")
        record_video("BipedalWalker-v3", model, video_filename, strength_ratio=0.3)

    # 3. Save final model
    print(f"Saving final adapted model to {FINAL_MODEL_NAME}")
    model.save(FINAL_MODEL_NAME)
    train_env.close()
    print("Done!")
