import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
import sys
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from torchvision import transforms
from nets.FullVggCompositionNet import FullVggCompositionNet as CompositionNet
from nets.SiameseNet import SiameseNet
# from datasets import data_transforms

def get_val_transform():
    """ takes tensor input, assumes correct image_size (no cropping needed)"""
    val_transform = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    return val_transform

class VenArgs:
    def __init__(self):
        self.l1 = 1024
        self.l2 = 512
        self.gpu_id = 0
        self.multiGpu = True
        self.resume = ['/mnt/HDD3/miayan/omega/RL/gaussian-splatting/pretrain/model_params/EvaluationNet.pth.tar']

def init_aesthetic_model(device="cuda"):
    print("Initializing Aesthetic Model")
    ven_args = VenArgs()
    
    ckpt_file=ven_args.resume
    found = False
    if ckpt_file is not None:
        for f in ckpt_file:
            if os.path.isfile(f):
                ckpt_file = f
                found = True
                break
        if not found:
            print(f"Aesthetic Model {ckpt_file} does not exist, exiting")
            sys.exit(-1)
        #print("load from {:s}".format(ckpt_file))
        
        single_pass_net = CompositionNet(pretrained=False, LinearSize1=ven_args.l1, LinearSize2=ven_args.l2)
        siamese_net = SiameseNet(single_pass_net)
        ckpt = torch.load(ckpt_file, map_location=lambda storage, loc: storage)
        model_state_dict = ckpt['state_dict']
        siamese_net.load_state_dict(model_state_dict)  # this loads the weights of single_pass_net within Siamese Net
    else:
        single_pass_net=CompositionNet(pretrained=True, LinearSize1=ven_args.l1, LinearSize2=ven_args.l2)
        siamese_net=SiameseNet(single_pass_net)
    #print("Number of Params in {:s}\t{:d}".format(identifier, sum([p.data.nelement() for p in single_pass_net.parameters()])))
    
    
    if torch.cuda.device_count()>0:
        # ven_args.gpu_id = 0
        # torch.cuda.set_device(int(ven_args.gpu_id))
        # single_pass_net.cuda()
        # cudnn.benchmark = True  # https://discuss.pytorch.org/t/what-does-torch-backends-cudnn-benchmark-do/5936
        single_pass_net.to(device)
    else:
        ven_args.gpu_id = None
    
    single_pass_net.eval()
    
    val_transform = get_val_transform()
    return single_pass_net, val_transform


class AestheticsModel:
    def __init__(self, negative_reward=-10., device="cuda"):
        self.single_pass_net, self.val_transform = init_aesthetic_model(device)
        self.negative_reward = negative_reward
        self.device = device
        # self.negative_reward = None  # disable negative reward
    
    def ImageScore(self, data):
        """ assumes data is 1*3*240*240 NCHW float tensor on GPU, range [0,1]"""
        with torch.no_grad():
            black_mask = (data.detach().cpu().numpy() < 1e-4).all(axis=0)   # shape (H,W) bool
            black_ratio = black_mask.mean()         # float
            if black_ratio >= 0.40:
                return self.negative_reward
            
            data = data.to(self.device)
            t_image_crop = self.val_transform(data)
            t_output = self.single_pass_net(t_image_crop)
            imagescore = t_output.data.cpu().numpy()[0][0]
        
        return imagescore
    
    def __call__(self, img):
        """ assumes pos is np adarray, img is torch sensor on GPU"""
        imgscore = self.ImageScore(img)  # imgscore is a np ndarray
        reward = imgscore
        return imgscore
