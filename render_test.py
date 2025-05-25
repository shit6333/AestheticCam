import os
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
import torch
import numpy as np
from tqdm import tqdm
from os import makedirs
from gaussian_renderer import render
import torchvision
from argparse import ArgumentParser
from rl_utils.utils_cam import MiniCam, Camera
from rl_utils.utils_pipe import DummyPipeline
from rl_utils.utils_gaussian import GaussianModel
from kornia.geometry.linalg import compose_transformations

def load_traj(filepath: str) -> np.ndarray:
    try:
        # Load trajectory file with arbitrary whitespace separators
        data = np.loadtxt(filepath, dtype=np.float64)
    except Exception as e:
        raise RuntimeError(f"Failed to read file: {e}")
    
    return data

def random_camera_pose_normal(
    gaussian_xyz: torch.Tensor,
    yaw_range: float = np.pi,
    pitch: float = 0.0,
    var_scale: float = 0.8
):
    """
    Randomly generate camera extrinsic (R, T) based on the distribution of gaussian_xyz:
      - T ~ N(mean_xyz, var_xyz * var_scale)
      - yaw ∈ [-yaw_range, yaw_range] sampled uniformly
      - pitch is fixed (roll = 0)
    """
    xyz = gaussian_xyz.detach().cpu().numpy()
    mean_xyz = xyz.mean(axis=0)
    var_xyz  = xyz.var(axis=0) * var_scale
    T = np.random.normal(loc=mean_xyz, scale=np.sqrt(var_xyz)).astype(np.float32)

    yaw = np.random.uniform(-yaw_range, yaw_range)
    cy, sy = np.cos(0.0), np.sin(0.0)
    cp, sp = np.cos(pitch), np.sin(pitch)
    R_x = np.array([[1,0,0],[0,cp,-sp],[0,sp,cp]], dtype=np.float32)
    R_y = np.array([[cy,0,sy],[0,1,0],[-sy,0,cy]], dtype=np.float32)
    R = R_y @ R_x
    return R, T

if __name__ == '__main__':
    
    # Load GS model
    # pretrain_ply_path = '/mnt/HDD3/miayan/omega/RL/gaussian-splatting/output/room/params.ply'
    pretrain_ply_path = '/mnt/HDD3/miayan/omega/RL/gaussian-splatting/output/91f6be8c-b/point_cloud/iteration_30000/point_cloud.ply'
    sh_degree = 3
    try:
        gaussian = GaussianModel(sh_degree=sh_degree)
        gaussian.load_ply(pretrain_ply_path)
        gaussian.to_cuda()
        print("Successfully loaded pretrained model from '{}'.".format(pretrain_ply_path))
    except Exception as e:
        print("Failed to load .ply file:", e)

    print(gaussian.get_features.shape)
    print(gaussian._features_dc.shape)
    print(gaussian._features_rest.shape)
    print(gaussian.get_opacity.shape)
    print(gaussian.get_xyz.shape)
    print(gaussian.get_scaling.shape)
    print(gaussian.get_rotation.shape)
    
    # Load camera trajectory (unused later, overridden by random sampling)
    pose_datas = np.array([[ # placeholder trajectory matrix
        -3.20569622e-01, 4.48055195e-01, -8.34554767e-01, 3.45298742e+00,
         9.47224956e-01, 1.51635452e-01, -2.82438617e-01, 4.54611013e-01,
         1.07897793e-16, -8.81052344e-01, -4.73018782e-01, 5.93628545e-01,
         0.0, 0.0, 0.0, 1.0
     ], [
        1.0, 0.0, 0.0, 0.0, 
        0.0, 1.0, 0.0, 0.0,
        0.0, 0.0, 1.0, 0.0,
        0.0, 0.0, 0.0, 1.0
     ]])

    pose_data = np.array([[ 0.83741821,-0.16963292 ,0.51957233,-2.43503025],
                            [ 0.21721823  ,0.97561218 ,-0.03157704 ,1.78571239],
                            [-0.50154459  ,0.13930377  ,0.85384277, -3.19813755],
                            [ 0.          ,0.          ,0.         ,1.        ]])
    #  0.04724081 -1.70971143  4.0522685
    # pose_data = np.array([[1.0, 0.0, 0.0, 0.0], 
    #                       [0.0, 1.0, 0.0, 0.0],
    #                       [0.0, 0.0, 1.0, 0.0],
    #                       [0.0, 0.0, 0.0, 1.0]])

    pose_datas = load_traj('/mnt/HDD3/miayan/omega/SGS-SLAM/data/Replica/room0/traj.txt')

    render_path = "./temp/"
    background = torch.tensor([0, 0, 0], dtype=torch.float32, device="cuda")
    pipeline = DummyPipeline()
    
    # Intrinsic parameters
    width, height = 1557, 1038 # 3114, 2075 #  #  #1200, 680
    fx, fy = 1586, 1586 # 3173 # 600.0, 600.0
    fovx = 2 * np.arctan(width  / (2 * fx))
    fovy = 2 * np.arctan(height / (2 * fy))
    
    # Extrinsic parameters (pose)
    poses_c2w = pose_datas.reshape(-1, 4, 4)
    pose_c2w_ref = np.linalg.inv(poses_c2w[0])
    
    for i, pose_c2w_i in enumerate(poses_c2w):
        pose = compose_transformations(torch.from_numpy(pose_c2w_ref), torch.from_numpy(pose_c2w_i)).numpy()
        R = pose[:3, :3]
        T = pose[:3,  3]
        
        # Override pose with randomly sampled pose
        R, T = random_camera_pose_normal(gaussian.get_xyz, var_scale=0.7)
        
        pose_data_inv = np.linalg.inv(pose_data)
        R = pose_data[:3,:3]
        T = pose_data[:3, 3]
        R_inv = pose_data_inv[:3,:3]
        T_inv = pose_data_inv[:3, 3]
        
        print(R,T)
        print(R_inv,T_inv)
        T = T_inv
        # cam = Camera(R=R, T=T, W=width, H=height, FoVx=fovy, FoVy=fovx)
        cam = MiniCam(R=R, T=T, width=width, height=height,
                      fovy=fovy, fovx=fovx, znear=0.01, zfar=100.0)

        rendering = render(cam, gaussian, pipeline, background)["render"]
        torchvision.utils.save_image(rendering, os.path.join(render_path, f'render_{i}.png'))
        print(f'Saved image {i}')
        
        if i > 500:
            break
        break
