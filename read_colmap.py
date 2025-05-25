import os
import struct
import numpy as np

def quaternion_to_rotation_matrix(qvec):
    """
    Convert quaternion [qw, qx, qy, qz] to a 3x3 rotation matrix.
    """
    qw, qx, qy, qz = qvec
    # Normalize quaternion
    n = np.linalg.norm(qvec)
    if n > 0.0:
        qw, qx, qy, qz = qw/n, qx/n, qy/n, qz/n
    R = np.array([
        [1 - 2*(qy**2 + qz**2),     2*(qx*qy - qz*qw),     2*(qx*qz + qy*qw)],
        [    2*(qx*qy + qz*qw), 1 - 2*(qx**2 + qz**2),     2*(qy*qz - qx*qw)],
        [    2*(qx*qz - qy*qw),     2*(qy*qz + qx*qw), 1 - 2*(qx**2 + qy**2)]
    ])
    return R


def read_cameras_binary(path):
    """
    Read COLMAP cameras.bin and return a dict of camera intrinsics.
    """
    model_id_to_name = {
        0: 'SIMPLE_PINHOLE', 1: 'PINHOLE', 2: 'SIMPLE_RADIAL',
        3: 'RADIAL', 4: 'OPENCV', 5: 'OPENCV_FISHEYE',
        6: 'FULL_OPENCV', 7: 'FOV', 8: 'SIMPLE_RADIAL_FISHEYE'
    }
    params_count = {
        'SIMPLE_PINHOLE':3, 'PINHOLE':4, 'SIMPLE_RADIAL':2,
        'RADIAL':3, 'OPENCV':8, 'OPENCV_FISHEYE':8,
        'FULL_OPENCV':12, 'FOV':5, 'SIMPLE_RADIAL_FISHEYE':2
    }
    cams = {}
    with open(path, 'rb') as f:
        num = struct.unpack('<Q', f.read(8))[0]
        for _ in range(num):
            cam_id = struct.unpack('<I', f.read(4))[0]
            model_id = struct.unpack('<I', f.read(4))[0]
            w = struct.unpack('<I', f.read(4))[0]
            h = struct.unpack('<I', f.read(4))[0]
            model = model_id_to_name.get(model_id, 'UNKNOWN')
            pc = params_count.get(model, 0)
            params = struct.unpack('<' + 'd'*pc, f.read(8*pc))
            cams[cam_id] = {
                'model': model,
                'width': w,
                'height': h,
                'params': np.array(params)
            }
    return cams


def read_images_binary(path):
    """
    Read COLMAP images.bin and return a dict of camera-to-world poses.
    """
    imgs = {}
    with open(path, 'rb') as f:
        num = struct.unpack('<Q', f.read(8))[0]
        for _ in range(num):
            img_id = struct.unpack('<I', f.read(4))[0]
            qvec = np.array(struct.unpack('<4d', f.read(32)))
            tvec = np.array(struct.unpack('<3d', f.read(24)))
            cam_id = struct.unpack('<I', f.read(4))[0]
            # read name
            name_bytes = []
            while True:
                c = f.read(1)
                if c == b'\x00': break
                name_bytes.append(c)
            name = b''.join(name_bytes).decode('utf-8')
            # skip 2D points
            n_pts2d = struct.unpack('<Q', f.read(8))[0]
            f.read(n_pts2d * (8+8+8))
            # compute C2W
            R_wc = quaternion_to_rotation_matrix(qvec)
            R_cw = R_wc.T
            t_cw = -R_cw.dot(tvec)
            H = np.eye(4)
            H[:3,:3] = R_cw
            H[:3, 3] = t_cw
            imgs[img_id] = {
                'name': name,
                'camera_id': cam_id,
                'c2w': H}
    return imgs


def main(model_dir):
    cam_file = os.path.join(model_dir, 'cameras.bin')
    img_file = os.path.join(model_dir, 'images.bin')
    cameras = read_cameras_binary(cam_file)
    images = read_images_binary(img_file)

    # choose first image
    first_id = sorted(images.keys())[0]
    img_info = images[first_id]
    cam_info = cameras[img_info['camera_id']]

    # print camera intrinsics
    print(f"Camera Intrinsics (ID {img_info['camera_id']}):")
    print(f"  Model: {cam_info['model']}")
    print(f"  Resolution: {cam_info['width']} x {cam_info['height']}")
    print(f"  Params: {cam_info['params']}")

    # print C2W pose
    print(f"\nImage {first_id} ('{img_info['name']}') C2W Pose (4x4):")
    print(img_info['c2w'])

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_dir', type=str, default='/mnt/HDD3/miayan/omega/RL/gaussian-splatting/data/room/sparse/0',help='Directory containing cameras.bin and images.bin')
    args = parser.parse_args()
    main(args.model_dir)
