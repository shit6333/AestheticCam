import os
from pathlib import Path

import torch
from torchvision.utils import save_image

from gaussian_rl_env import GaussianRLSceneEnv

def run_demo(
    ply_path: str,
    out_dir: str = "./demo_frames",
    img_width: int = 800,
    img_height: int = 600,
    max_steps: int = 200,
    seed: int = 0,
):
    """
    Run a single episode using a random policy and save each rendered frame to out_dir.
    """
    # Ensure output directory exists
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    # Create the environment
    env = GaussianRLSceneEnv(
        ply_path=plyy_path,
        img_width=img_width,
        img_height=img_height,
        max_steps=max_steps,
    )

    # Reset the environment
    obs, _ = env.reset(seed=seed)
    frame_idx = 0

    # Save the initial frame
    save_image(obs, f"{out_dir}/frame_{frame_idx:04d}.png")

    done = False
    while not done:
        frame_idx += 1

        # Sample a random action (replace with your own policy as needed)
        action = env.action_space.sample()

        # Step the environment
        obs, reward, terminated, truncated, _ = env.step(action)

        # Save the observation; obs shape = (3, H, W), values in [0, 1]
        save_image(obs, f"{out_dir}/frame_{frame_idx:04d}.png")

        # Check termination
        done = terminated or truncated

    # Close the environment
    env.close()
    print(f"✓ Saved {frame_idx + 1} frames to '{out_dir}'")


if __name__ == "__main__":
    # Replace the path with your 3DGS parameters .ply file
    PLY_PATH = '/mnt/HDD3/miayan/omega/RL/gaussian-splatting/output/room/params.ply'
    run_demo(
        ply_path=PLY_PATH,
        out_dir="./demo_frames",
        img_width=1200,
        img_height=680,
        max_steps=20,
        seed=9,
    )
