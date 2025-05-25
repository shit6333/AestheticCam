import argparse
import os
import numpy as np
import torch
from collections import deque
from env import UnifiedGaussianEnv
# from agent import DrQV2Agent
from drqv2_net import DrQV2Agent
from aesthetics_model import AestheticsModel
from torch.utils.tensorboard import SummaryWriter

def parse_args():
    parser = argparse.ArgumentParser(description="Train DrQV2Agent on UnifiedGaussianEnv")
    parser.add_argument("--ply", type=str, default="/mnt/HDD3/miayan/omega/RL/gaussian-splatting/output/room_midnerf/point_cloud/iteration_30000/point_cloud.ply", help="PLY file for the scene parameters")
    parser.add_argument("--device", type=str, default="cuda", help="PyTorch device")
    parser.add_argument("--episodes", type=int, default=100000, help="Number of training episodes")
    parser.add_argument("--max_steps", type=int, default=200, help="Max steps per episode")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate for agent optimizers")
    parser.add_argument("--buffer_size", type=int, default=100000, help="Replay buffer capacity")
    parser.add_argument("--batch_size", type=int, default=256, help="Batch size for updates")
    parser.add_argument("--start_steps", type=int, default=2000, help="Steps to populate buffer with random policy before updates")
    parser.add_argument("--update_after", type=int, default=2000, help="Steps after which to start updating")
    parser.add_argument("--update_every", type=int, default=2, help="Update agent every N environment steps")
    parser.add_argument("--save_interval", type=int, default=2000, help="Save model every N episodes")
    parser.add_argument("--outdir", type=str, default="./checkpoints", help="Directory to save agent checkpoints")
    return parser.parse_args()


class ReplayBuffer:
    def __init__(self, obs_shape, pose_dim, action_dim, 
                 size, device, 
                 history_shape=None, excluding_shape=None,
                 gamma=0.99):
        self.device = device
        self.max_size = size
        self.ptr = 0
        self.size = 0
        self.gamma = gamma
        # observations: image and pose and time
        self.obs_img = np.zeros((size, *obs_shape), dtype=np.uint8)
        self.obs_pose = np.zeros((size, pose_dim), dtype=np.float32)
        self.obs_t = np.zeros((size, 1), dtype=np.int32)
        self.actions = np.zeros((size, action_dim), dtype=np.float32)
        self.rewards = np.zeros((size, 1), dtype=np.float32)
        self.done = np.zeros((size, 1), dtype=np.float32)
        self.next_obs_img = np.zeros((size, *obs_shape), dtype=np.uint8)
        self.next_obs_pose = np.zeros((size, pose_dim), dtype=np.float32)
        self.next_obs_t = np.zeros((size, 1), dtype=np.int32)

        self.history = None
        self.next_history = None
        self.excluding = None
        self.next_excluding = None
        if history_shape is not None:
            self.history = np.zeros((size, *history_shape), dtype=np.float32)
            self.next_history = np.zeros((size, *history_shape), dtype=np.float32)
        if excluding_shape is not None:
            self.excluding = np.zeros((size, *excluding_shape), dtype=np.float32)
            self.next_excluding = np.zeros((size, *excluding_shape), dtype=np.float32)

    def add(self, obs, pose, t, action, reward, done, next_obs, next_pose, next_t, 
            history=None, excluding=None, next_history=None, next_excluding=None):
        i = self.ptr
        self.obs_img[i] = (obs * 255).astype(np.uint8) if obs.dtype==np.float32 else obs
        self.obs_pose[i] = pose
        self.obs_t[i] = t
        self.actions[i] = action
        self.rewards[i] = reward
        self.done[i] = done
        self.next_obs_img[i] = (next_obs * 255).astype(np.uint8) if next_obs.dtype==np.float32 else next_obs
        self.next_obs_pose[i] = next_pose
        self.next_obs_t[i] = next_t
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

        if history is not None:
            self.history[i] = history
            self.next_history[i] = next_history
        if excluding is not None:
            self.excluding[i] = excluding
            self.next_excluding[i] = next_excluding

    def sample(self, batch_size):
        idxs = np.random.randint(0, self.size, size=batch_size)
        obs = torch.as_tensor(self.obs_img[idxs], device=self.device).float() / 255.0
        pose = torch.as_tensor(self.obs_pose[idxs], device=self.device)
        t = torch.as_tensor(self.obs_t[idxs], device=self.device).float()
        action = torch.as_tensor(self.actions[idxs], device=self.device)
        reward = torch.as_tensor(self.rewards[idxs], device=self.device)
        done = torch.as_tensor(self.done[idxs], device=self.device)
        next_obs = torch.as_tensor(self.next_obs_img[idxs], device=self.device).float() / 255.0
        next_pose = torch.as_tensor(self.next_obs_pose[idxs], device=self.device)
        next_t = torch.as_tensor(self.next_obs_t[idxs], device=self.device).float()
        discount = (1.0 - done) * self.gamma
        # Return tuple matching agent.update signature
        history = None
        next_history = None
        excluding = None
        next_excluding = None
        if self.history is not None:
            history = torch.as_tensor(self.history[idxs], device=self.device)
            next_history = torch.as_tensor(self.next_history[idxs], device=self.device)
        if self.excluding is not None:
            excluding = torch.as_tensor(self.excluding[idxs], device=self.device)
            next_excluding = torch.as_tensor(self.next_excluding[idxs], device=self.device)

        return obs, pose, t, action, reward, discount, next_obs, next_pose, next_t, history, next_history, excluding, next_excluding


def train():
    args = parse_args()
    os.makedirs(args.outdir, exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

    # tensor board
    tb_log_dir = os.path.join(args.outdir, "tb_logs")
    writer = SummaryWriter(log_dir=tb_log_dir)

    # build env
    aes_model = AestheticsModel(device=device)
    env = UnifiedGaussianEnv(
        ply_path=args.ply,
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

    # build agent
    obs_shape = env.observation_space['image'].shape
    # print(f"Obs Shape : {obs_shape}")
    pose_shape = [5]
    # print(f"Pose Shape : {pose_shape}")
    history_shape = (3,5) # env.observation_space['history_poses'].shape
    excluding_shape = (3,5)
    pose_dim = pose_shape[0]
    action_shape = env.action_space.shape
    action_dim = action_shape[0]
    gamma = 0.99

    agent = DrQV2Agent(
        obs_shape=obs_shape,
        pos_shape=pose_shape,
        action_shape=action_shape,
        device=device,
        lr=args.lr,
        feature_dim=50,
        hidden_dim=1024,
        critic_target_tau=0.01,
        num_expl_steps=10000,
        update_every_steps=args.update_every,
        stddev_schedule='linear(1.0,0.1,1000000)',
        stddev_clip=0.3,
        use_tb=False,
        use_context=False,
        context_hidden_dim=128,
        context_history_length=5,
        nstep=1,
        batch_size=args.batch_size,
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
        position_orientation_separate=False, # True,
        rand_diversity_radius=False,
        constant_noise=-1,
        no_aug=False,
    )

    # build replay buffer
    buffer = ReplayBuffer(obs_shape, pose_dim, action_dim, args.buffer_size, 
                          device, history_shape, excluding_shape, gamma=gamma)
    replay_iter = iter(lambda: buffer.sample(args.batch_size), None)

    total_steps = 0
    ewma_reward = 0
    for ep in range(1, args.episodes + 1):
        
        # reset env
        obs_dict, _ = env.reset()
        obs_img = obs_dict['image']
        obs_pose = obs_dict['pose'][:-1]
        history_poses = obs_dict['history_poses'][:,:-1]
        excluding_poses = obs_dict['excluding_poses'][:,:-1]
        done = False
        ep_reward = 0.0
        t_step = 0
        aesthetic_avg = 0

        while not done and t_step < args.max_steps:
            # sample action
            if total_steps < args.start_steps:
                action = env.action_space.sample()
            else:
                action = agent.act(
                    obs=obs_img,
                    pos=obs_pose,
                    t=t_step,
                    excluding_seq=excluding_poses,
                    avg_step_size=history_poses,
                    step=total_steps,
                    eval_mode=False,
                    #history=history_poses
                )

            # interact with envirionment
            next_dict = env.step(action)
            next_obs_dict, reward, terminated, truncated, info = next_dict
            next_obs_img = next_obs_dict['image']
            next_obs_pose = next_obs_dict['pose'][:-1]
            next_history_poses = next_obs_dict['history_poses'][:,:-1]
            next_excluding_poses = next_obs_dict['excluding_poses'][:,:-1]
            done_flag = float(terminated or truncated)

            # save to Replay Buffer (obs, pose, t, action, reward, done, next_obs, next_pose, next_t)
            buffer.add(
                obs_img, obs_pose, np.array([t_step]),
                action, np.array([reward]), done_flag,
                next_obs_img, next_obs_pose, np.array([t_step+1]), 
                history_poses, excluding_poses, next_history_poses, next_excluding_poses
            )

            obs_img = next_obs_img
            obs_pose = next_obs_pose
            obs_dict = next_obs_dict
            ep_reward += reward
            aesthetic_avg += info['aesthetic']
            done = terminated or truncated
            t_step += 1
            total_steps += 1

            # update agent
            if total_steps >= args.update_after:
                agent.update(replay_iter, total_steps)

        ewma_reward = 0.05 * ep_reward + (1 - 0.05) * ewma_reward
        aesthetic_avg = aesthetic_avg / t_step
        print(f"Episode {ep} | Steps: {t_step} | Reward: {ep_reward:.2f} | EWMA Reward {ewma_reward:.2f} | Aes {aesthetic_avg:.2f}")

        # set tensorboard
        writer.add_scalar('Episode/Reward', ep_reward, ep)
        writer.add_scalar('Episode/EWMA_Reward', ewma_reward, ep)
        writer.add_scalar('Episode/Aesthetic', aesthetic_avg, ep)
        writer.flush()

        # save model
        if ep % args.save_interval == 0:
            ckpt_path = os.path.join(args.outdir, f"agent_ep{ep}.pth")
            torch.save({
                'agent_state_dict': agent.get_weights(),
                'step': total_steps,
            }, ckpt_path)
            print(f"Saved checkpoint: {ckpt_path}")

    env.close()
    writer.close()


if __name__ == '__main__':
    train()
