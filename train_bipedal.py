import os
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env

def record_video(model, step_label, video_dir="./videos"):
    """
    Records a single episode of the environment with the given model.
    Uses rgb_array for headless environments and RecordVideo wrapper.
    """
    os.makedirs(video_dir, exist_ok=True)

    # Create env for recording - must be rgb_array for headless environments (no GUI)
    env = gym.make("BipedalWalker-v3", render_mode="rgb_array")
    
    from gymnasium.wrappers import RecordVideo
    
    # We want to record exactly one episode at this stage.
    # Using a trigger that returns True when the first episode is completed. 
    # In Gymnasium, we can just use a custom trigger or let it default and close after one step if needed.
    # Spec: "episode_trigger=lambda x: x == 0" - Wait, this means on the very FIRST attempt of an episode?
    # Actually RecordVideo uses 'x' which is typically episode index/number in some implementations.
    # In gymnasium.wrappers.RecordVideo (current version), it checks if episode_trigger(episode) returns True. 
    # If we want to record the first episode that happens:

    env = RecordVideo(
        env, 
        video_folder=video_dir, 
        name_prefix=f"step_{step_label}",
        episode_trigger=lambda x: True  # We'll just force it for this single run.
    )

    obs, info = env.reset()
    terminated = False
    truncated = False
    
    steps = 0
    max_steps = 1600 # Spec says "最大1600ステップ実行"
    
    while not (terminated or truncated) and steps < max_steps:
        if model is None:
            action, _states = env.action_space.sample(), {}
        else:
            action, _states = model.predict(obs, deterministic=True)
        
        # Gymnasium step returns 5 values (next_state, reward, terminated, truncated, info)
        obs, reward, terminated, truncated, info = env.step(action)
        steps += 1

    env.close()
    print(f"Finished recording video for step_{step_label}")

if __name__ == "__main__":
    # Create directories
    os.makedirs("./videos", exist_ok=True)
    os.makedirs("./models", exist_ok=True)
    os.makedirs("./tb_logs", exist_ok=True)

    # 1. Initial recording (0k steps, model is None)
    print("Recording initial video (000k)...")
    record_video(model=None, step_label="000k")

    # 2. Setup training environment - 4 parallel environments as per spec
    n_envs = 4
    train_env = make_vec_env("BipedalWalker-v3", n_envs=n_envs)

    # 3. Initialize PPO model with TensorBoard logging
    print("Initializing PPO model...")
    model = PPO(
        "MlpPolicy", 
        train_env, 
        verbose=1, 
        tensorboard_log="./tb_logs/"
    )

    # 4. Training loop: total 500k steps in 100k increments (total of 5 iterations)
    # We need to track current 'total' for the step label as requested.
    current_training_step = 0
    increment = 100_000

    print("Starting training...")
    for i in range(5): # 5 intervals of 100k each
        # Spec: model.learn(total_timesteps=100_000, reset_num_timesteps=False, tb_log_name="PPO_Bipedal")
        model.learn(total_timesteps=increment, reset_num_timesteps=False, tb_log_name="PPO_Bipedal")
        current_training_step += increment
        print(f"Completed {current_training_step} steps.")
        
        # Record video at this milestone: f"{total // 1000}k" where total is the current training step.
        record_video(model=model, step_label=f"{current_training_step // 1000}k")

    # Save model and close env
    model.save("./models/ppo_bipedal_500k")
    train_env.close()
    print("Training completed successfully.")
