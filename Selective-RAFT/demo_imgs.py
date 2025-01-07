import sys
sys.path.append('core')
import os,cv2
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import argparse
import glob
import numpy as np
import torch
from tqdm import tqdm
from pathlib import Path
from core.raft import RAFT
from utils.utils import InputPadder
from PIL import Image
from matplotlib import pyplot as plt
import torch.nn.functional as F

DEVICE = 'cuda'

def load_image(imfile):
    img = np.array(Image.open(imfile)).astype(np.uint8)[..., :3]
    img = torch.from_numpy(img).permute(2, 0, 1).float()
    return img[None].to(DEVICE)

def demo(args):
    model = torch.nn.DataParallel(RAFT(args), device_ids=[0])
    model.load_state_dict(torch.load(args.restore_ckpt))

    model = model.module
    model.to(DEVICE)
    model.eval()

    output_directory = Path(args.output_directory)
    output_directory.mkdir(exist_ok=True)

    with torch.no_grad():
        left_images = sorted(glob.glob(args.left_imgs, recursive=True))
        right_images = sorted(glob.glob(args.right_imgs, recursive=True))
        print(f"Found {len(left_images)} images. Saving files to {output_directory}/")

        for (imfile1, imfile2) in tqdm(list(zip(left_images, right_images))):
            image1 = load_image(imfile1)
            image2 = load_image(imfile2)

            left_cv_image = cv2.imread(imfile1)
            right_cv_image = cv2.imread(imfile1)

            padder = InputPadder(image1.shape, divis_by=32)
            image1, image2 = padder.pad(image1, image2)

            _, disp = model(image1, image2, iters=args.valid_iters, test_mode=True)

            disp = -disp.cpu().numpy()
            disp = padder.unpad(disp)
            file_stem = imfile1.split('/')[-2]
            filename = os.path.join(output_directory, f'{file_stem}.png')
            # plt.imsave(filename, disp.squeeze(), cmap='jet')
            file_stem = imfile1.split('/')[-1]
            name,end = file_stem.split(".")
            filename = os.path.join(output_directory, name+"_depth."+end )

            # 合并显示
            disp = np.round(disp * 256).astype(np.uint16)
            depth = cv2.applyColorMap(cv2.convertScaleAbs(disp.squeeze(), alpha=0.01),cv2.COLORMAP_JET)
            depth = np.append(np.append(left_cv_image,right_cv_image,1),depth,1)

            cv2.rectangle(depth,(0,0),  (160,70),(0,255,0),-1)
            cv2.rectangle(depth,(640,0),(640+200,70),(0,255,0),-1)
            cv2.rectangle(depth,(640+640,0),(640+640+210,70),(0,255,0),-1)
            cv2.putText(depth,"left",  (20, 50), 5,3, (255,0,255),3)
            cv2.putText(depth,"right", (0+640,50),5,3,(255,0,255),3)
            cv2.putText(depth,"depth", (0+640+640,50),5,3,(255,0,255),3)
            ih,iw,ic = depth.shape
            cv2.line(depth,(int(iw/3),0),(int(iw/3),ih),(255,0,128),4,1 )
            cv2.line(depth,(int(iw/3)*2,0),(int(iw/3)*2,ih),(255,0,128),4,2 )

            cv2.imwrite(filename,depth , [int(cv2.IMWRITE_PNG_COMPRESSION), 0] )

            cv2.imshow("depth",depth)
            cv2.waitKey(1)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--restore_ckpt', default=None, help="restore checkpoint")
    parser.add_argument('--save_numpy', action='store_true', help='save output as numpy arrays')
    parser.add_argument('-l', '--left_imgs', help="path to all first (left) frames", default=None)
    parser.add_argument('-r', '--right_imgs', help="path to all second (right) frames", default=None)
    parser.add_argument('--output_directory', help="directory to save output", default=None)
    parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')
    parser.add_argument('--valid_iters', type=int, default=32, help='number of flow-field updates during forward pass')

    # Architecture choices
    parser.add_argument('--hidden_dim', nargs='+', type=int, default=128, help="hidden state and context dimensions")
    parser.add_argument('--corr_implementation', choices=["reg", "alt", "reg_cuda", "alt_cuda"], default="reg", help="correlation volume implementation")
    parser.add_argument('--shared_backbone', action='store_true', help="use a single backbone for the context and feature encoders")
    parser.add_argument('--corr_levels', type=int, default=4, help="number of levels in the correlation pyramid")
    parser.add_argument('--corr_radius', type=int, default=4, help="width of the correlation pyramid")
    parser.add_argument('--n_downsample', type=int, default=2, help="resolution of the disparity field (1/2^K)")
    parser.add_argument('--slow_fast_gru', action='store_true', help="iterate the low-res GRUs more frequently")
    parser.add_argument('--n_gru_layers', type=int, default=3, help="number of hidden GRU levels")
    parser.add_argument('--precision_dtype',type=str, default='float16', choices=['float16', 'bfloat16', 'float32'], help='Choose precision type: float16 or bfloat16 or float32')

    args = parser.parse_args()

    demo(args)
