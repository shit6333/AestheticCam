import torch
import numpy as np
from plyfile import PlyData

def load_ply_and_print_shapes(ply_path):
    """
    Load a .ply file and print the key and tensor shape for each field.
    """
    plydata = PlyData.read(ply_path)
    # Get vertex data from the ply file, which is a numpy structured array
    vertex_data = plydata['vertex'].data
    print("Fields found in file:", vertex_data.dtype.names)
    
    for key in vertex_data.dtype.names:
        # Retrieve data for a single field (as a numpy array)
        array_value = vertex_data[key]
        # Convert to torch tensor
        tensor_value = torch.tensor(array_value)
        print(f"Key: '{key}', Tensor shape: {tensor_value.shape}")
    
    print("cyz")
    x = torch.tensor(vertex_data['x'])
    y = torch.tensor(vertex_data['y'])
    z = torch.tensor(vertex_data['z'])
    xyz = torch.stack([x, y, z], dim=1)
    print(xyz)
    print(xyz.shape)
    print(torch.min(x, dim=0).values, torch.max(x, dim=0).values)
    print(torch.min(y, dim=0).values, torch.max(y, dim=0).values)
    print(torch.min(z, dim=0).values, torch.max(z, dim=0).values)
    print("\n")
    
    print("Scale")
    x = torch.tensor(vertex_data['scale_0'])
    y = torch.tensor(vertex_data['scale_1'])
    z = torch.tensor(vertex_data['scale_2'])
    xyz = torch.stack([x, y, z], dim=1)
    print(xyz)
    print(xyz.shape)
    
    print('Opacity')
    print(vertex_data['opacity'])
    print(vertex_data['opacity'].shape)
    
    print('Rotation')
    x = torch.tensor(vertex_data['rot_0'])
    y = torch.tensor(vertex_data['rot_1'])
    z = torch.tensor(vertex_data['rot_2'])
    g = torch.tensor(vertex_data['rot_3'])
    xyz = torch.stack([x, y, z, g], dim=1)
    print(xyz)
    print(xyz.shape)

if __name__ == '__main__':
    # Set the path to the ply file; modify according to your environment
    ply_file_path = '/mnt/HDD6/miayan/omega/SplatFormer2/test_custom3/nv3d_image_train_gs/coffee_martini/frame_00000_0/point_cloud/iteration_3000/point_cloud.ply'
    ply_file_path = '/mnt/HDD6/miayan/omega/gaussian-splatting/output/frame_00000/point_cloud/iteration_10000/point_cloud.ply'
    # ply_file_path = '/mnt/HDD3/miayan/omega/RL/gaussian-splatting/output/b98c8a6d-2/point_cloud/iteration_30000/point_cloud.ply'
    ply_file_path = '/mnt/HDD3/miayan/omega/RL/gaussian-splatting/output/room/params.ply'
    ply_file_path = '/mnt/HDD3/miayan/omega/RL/gaussian-splatting/output/fe421606-c/point_cloud/iteration_7000/point_cloud.ply'
    load_ply_and_print_shapes(ply_file_path)
