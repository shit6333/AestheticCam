import os
from pathlib import Path
import torch
from torchvision.utils import save_image
from env import UnifiedGaussianEnv
from aesthetics_model import AestheticsModel
from drqv2_net import DrQV2Agent

@torch.no_grad()
def test_agent(
    ply_path: str,
    ckpt_path: str,
    out_dir: str = "./demo_test_frames",
    max_steps: int = 200,
    init_idx: int = -1,
    device: str = "cuda"
):
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    # build environment
    aes_model = AestheticsModel(device=device)
    env = UnifiedGaussianEnv(
        ply_path=ply_path,
        aes_model=aes_model,
        img_width=1557,
        img_height=1038,
        fx=1586.0,
        fy=1586.0,
        smooth_window=3,
        history_length=3,
        excluding_length=3,
        device=device,
        sh_degree=3,
        images_bin = '/mnt/HDD3/miayan/omega/RL/gaussian-splatting/data/room/sparse/0/images.bin' # colmap images.bin path
    )

    obs_shape = env.observation_space["image"].shape
    pose_shape = [5]
    action_shape = env.action_space.shape

    # init agent & load weights
    agent = DrQV2Agent(
        obs_shape=obs_shape,
        pos_shape=pose_shape,
        action_shape=action_shape,
        device=device,
        lr=1e-4,
        feature_dim=50,
        hidden_dim=1024,
        critic_target_tau=0.01,
        num_expl_steps=10000,
        update_every_steps=2,
        stddev_schedule='linear(1.0,0.1,1000000)',
        stddev_clip=0.3,
        use_tb=False,
        use_context=False,
        context_hidden_dim=128,
        context_history_length=5,
        nstep=1,
        batch_size=256,
        num_scenes=1,
        use_position=True,
        diversity=True,
        exc_hidden_size=128,
        no_hidden=False,
        num_excluding_sequences=env.excluding_length,
        order_invariant=False,
        distance_obs=False,
        smoothness=True,
        position_only_smoothness=False,
        smoothness_window=3,
        position_orientation_separate=False,
        rand_diversity_radius=False,
        constant_noise=-1,
        no_aug=False,
    )

    # load checkpoint
    ckpt = torch.load(ckpt_path, map_location=device)
    agent.set_weights(ckpt['agent_state_dict'])

    # testing
    if init_idx != -1:
        obs_dict, _ = env.reset(idx=init_idx)
    else:
        obs_dict, _ = env.reset()
    obs_img = obs_dict['image']
    obs_pose = obs_dict['pose'][:-1]
    history_poses = obs_dict['history_poses'][:, :-1]
    excluding_poses = obs_dict['excluding_poses'][:, :-1]

    frame_idx = 0
    vis_img = env._render()
    save_image(torch.from_numpy(vis_img), f"{out_dir}/frame_{frame_idx:04d}.png")

    done = False
    t_step = 0
    while not done and t_step < max_steps:
        action = agent.act(
            obs=obs_img,
            pos=obs_pose,
            t=t_step,
            excluding_seq=excluding_poses,
            avg_step_size=history_poses,
            step=-1, 
            eval_mode=True
        )
        # action = env.action_space.sample() # random sample

        obs_dict, _, terminated, truncated, _ = env.step(action)
        obs_img = obs_dict['image']
        obs_pose = obs_dict['pose'][:-1]
        history_poses = obs_dict['history_poses'][:, :-1]
        excluding_poses = obs_dict['excluding_poses'][:, :-1]

        frame_idx += 1
        vis_img = env._render()
        save_image(torch.from_numpy(vis_img), f"{out_dir}/frame_{frame_idx:04d}.png")

        done = terminated or truncated
        t_step += 1
        print(f'Step: {t_step} save to => f"{out_dir}/frame_{frame_idx:04d}.png"')

    print(f"Test finished. Total steps: {t_step}")
    env.close()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--ply", type=str, default="/mnt/HDD3/miayan/omega/RL/gaussian-splatting/output/room_midnerf/point_cloud/iteration_30000/point_cloud.ply", help="Path to .ply file")
    parser.add_argument("--ckpt", type=str, default="/mnt/HDD3/miayan/omega/RL/gaussian-splatting/checkpoints/agent_ep40000.pth", help="Path to model checkpoint (.pth)")
    parser.add_argument("--outdir", type=str, default="./demo_test_frames")
    parser.add_argument("--max_steps", type=int, default=20)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    test_agent(
        ply_path=args.ply,
        ckpt_path=args.ckpt,
        out_dir=args.outdir,
        max_steps=args.max_steps,
        device=args.device,
        init_idx = 10
    )
