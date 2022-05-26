# Core Imports
import time
import argparse
import random
import os
from tqdm import tqdm
import glob
# External Dependency Imports
from imageio import imwrite
import torch
import numpy as np

# Internal Project Imports
from pretrained.vgg import Vgg16Pretrained, EffNet
from utils import misc as misc
from utils.misc import load_path_for_pytorch
from utils.stylize import produce_stylization

# Fix Random Seed
random.seed(0)
np.random.seed(0)
torch.manual_seed(0)

# Define command line parser and get command line arguments
parser = argparse.ArgumentParser()
parser.add_argument('--content_dir'   , type=str, 
    # default='/media/dereyly/svision_ssd2/ImageDB/face/imdb/imdb'
    default = '/media/dereyly/svision_ssd/ImageDB/face/Selfie-real/images'
    )
parser.add_argument('--style_dir'     , type=str, default='/media/dereyly/svision_ssd2/ImageDB/photo/styles1')
parser.add_argument('--output_dir'    , type=str, default='/media/dereyly/ssd4/ImageDB/xix/ST/10535_selfie')
parser.add_argument('--high_res'       , action='store_true'                  )
parser.add_argument('--cpu'            , action='store_true'                  )
parser.add_argument('--no_flip'        , action='store_true'                  )
parser.add_argument('--content_loss'   , action='store_true'                  )
parser.add_argument('--dont_colorize'  , action='store_true'                  )
parser.add_argument('--alpha'          , type=float, default=0.75             )
args = parser.parse_args()

# Interpret command line arguments
os.makedirs(args.output_dir,exist_ok=True)
# paths=os.listdir(args.content_dir)
content_imgs_paths = glob.glob(f'{args.content_dir}/*.jpg')
content_imgs_paths += glob.glob(f'{args.content_dir}/*.jpeg')
content_imgs_paths += glob.glob(f'{args.content_dir}/*.png')
content_imgs_paths += glob.glob(f'{args.content_dir}/**/*.jpg')

for content_path in  tqdm(content_imgs_paths[20000:]):
    content_name = content_path.split('/')[-1]
    for style_name in os.listdir(args.style_dir):
        # content_path = f'{args.content_dir}/{content_name}'
        style_path = f'{args.style_dir}/{style_name}'
        output_path = f'{args.output_dir}/{content_name}_{style_name}.jpg'
        max_scls = 4
        sz = 512
        if args.high_res:
            max_scls = 5
            sz = 1024
        flip_aug = (not args.no_flip)
        content_loss = args.content_loss
        misc.USE_GPU = (not args.cpu)
        content_weight = 1. - args.alpha

        # Error checking for arguments
        # error checking for paths deferred to imageio
        assert (0.0 <= content_weight) and (content_weight <= 1.0), "alpha must be between 0 and 1"
        assert torch.cuda.is_available() or (not misc.USE_GPU), "attempted to use gpu when unavailable"

        # Define feature extractor
        cnn = misc.to_device(Vgg16Pretrained()) #.half()
        # print(cnn)
        phi = lambda x, y, z: cnn.forward(x, inds=y, concat=z)

        # Load images
        content_im_orig = misc.to_device(load_path_for_pytorch(content_path, target_size=sz)).unsqueeze(0) #.half()
        style_im_orig = misc.to_device(load_path_for_pytorch(style_path, target_size=sz)).unsqueeze(0) #.half()

        # Run Style Transfer
        torch.cuda.synchronize()
        start_time = time.time()
        # try:
        output = produce_stylization(content_im_orig, style_im_orig, phi,
                                    max_iter=120,
                                    lr=2e-3,
                                    content_weight=content_weight,
                                    max_scls=max_scls,
                                    flip_aug=flip_aug,
                                    content_loss=content_loss,
                                    dont_colorize=args.dont_colorize)
        # except:
        #     print(content_path)
        #     continue
        torch.cuda.synchronize()
        print('Done! total time: {}'.format(time.time() - start_time))

        # Convert from pyTorch to numpy, clip to valid range
        new_im_out = np.clip(output[0].permute(1, 2, 0).detach().cpu().numpy(), 0., 1.)

        # Save stylized output
        save_im = (new_im_out * 255).astype(np.uint8)
        imwrite(output_path, save_im)

        # Free gpu memory in case something else needs it later
        if misc.USE_GPU:
            torch.cuda.empty_cache()
