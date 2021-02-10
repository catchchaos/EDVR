import os, torch, numpy as np
from torch.utils import data as data
from pathlib import Path

import imageio as io

class SingleVideoDataset(data.Dataset):
    def __init__(self, opt):
        super(SingleVideoDataset, self).__init__()
        self.opt = opt
        self.gt_root, self.lq_root = Path(opt['dataroot']), Path(
            opt['dataroot'])
        self.fps = 10	# Add to opt
        self.win_size = 10	# Add to opt

        lr_dir = os.path.join(opt['dataroot'], 'lr_frames', opt['name'])
        hr_dir = os.path.join(opt['dataroot'], 'hr_frames', opt['name'])
        num_lr = len(os.listdir(lr_dir)) - 1
        
        self.lr_idx = []
        self.hr_idx = []
        for i in range(1 + self.fps, num_lr - self.fps, 120 // self.fps):
            self.lr_idx.append([os.path.join(lr_dir, f'frame_{x:05d}.png')
                                for x in range(i - self.win_size, i + self.win_size + 1)])
            self.hr_idx.append(os.path.join(hr_dir, f'frame_{i:05d}.png'))

        # with open(opt['meta_info_file'], 'r') as fin:
        #     self.keys = [line.split(' ')[0] for line in fin]

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        lr_imgs_np = np.stack([io.imread(f)[:20,:].transpose(-1, 0, 1) for f in self.lr_idx[idx]])
        lr_images = torch.FloatTensor(lr_imgs_np)  / 255.
        hr_image = torch.FloatTensor(io.imread(self.hr_idx[idx])[:320,:].transpose(-1, 0, 1)) / 255.

        # key = os.path.basename(self.lr_idx[idx]).strip('.png')
        sample = {'gt': hr_image, 'lq': lr_images, 'key': 'None'}

        return sample

    def __len__(self):
        return len(self.hr_idx)


class SingleVideoAllDataset(data.Dataset):
    def __init__(self, opt):
        super(SingleVideoAllDataset, self).__init__()
        self.opt = opt
        self.gt_root, self.lq_root = Path(opt['dataroot']), Path(
            opt['dataroot'])
        self.fps = 10   # Add to opt
        self.win_size = 10  # Add to opt

        lr_dir = os.path.join(opt['dataroot'], 'lr_frames', opt['name'])
        hr_dir = os.path.join(opt['dataroot'], 'hr_frames', opt['name'])
        num_lr = len(os.listdir(lr_dir)) - 1
        
        self.lr_idx = []
        self.hr_idx = []
        for i in range(1 + self.fps, num_lr - self.fps):
            self.lr_idx.append([os.path.join(lr_dir, f'frame_{x:05d}.png')
                                for x in range(i - self.win_size, i + self.win_size + 1)])
            self.hr_idx.append(os.path.join(hr_dir, f'frame_{i:05d}.png'))

        # with open(opt['meta_info_file'], 'r') as fin:
        #     self.keys = [line.split(' ')[0] for line in fin]

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        lr_imgs_np = np.stack([io.imread(f)[:20,:].transpose(-1, 0, 1) for f in self.lr_idx[idx]])
        lr_images = torch.FloatTensor(lr_imgs_np).contiguous()  / 255.
        hr_image = torch.FloatTensor(io.imread(self.hr_idx[idx])[:320,:].transpose(-1, 0, 1)) / 255.

        # key = os.path.basename(self.lr_idx[idx]).strip('.png')
        sample = {'gt': hr_image, 'lq': lr_images}

        return sample

    def __len__(self):
        return len(self.hr_idx)
