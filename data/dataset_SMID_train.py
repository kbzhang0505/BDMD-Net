import os.path as osp
import torch
import torch.utils.data as data
import data.util as util
import torch.nn.functional as F
import random
import cv2
import numpy as np
import glob
import os
import functools



class VideoSameSizeDataset(data.Dataset):
    def __init__(self, opt):
        super(VideoSameSizeDataset, self).__init__()
        self.opt = opt
        self.cache_data = opt['cache_data']
        self.half_N_frames = opt['N_frames'] // 2
        self.GT_root, self.LQ_root = opt['dataroot_GT'], opt['dataroot_LQ']
        self.data_type = self.opt['data_type']
        self.data_info = {'path_LQ': [], 'path_GT': [], 'folder': [], 'idx': [], 'border': []}
        if self.data_type == 'lmdb':
            raise ValueError('No need to use LMDB during validation/test.')
        # Generate data info and cache data
        self.imgs_LQ, self.imgs_GT = {}, {}

        testing_dir = []
        f = open('test_list.txt')
        lines = f.readlines()
        for mm in range(len(lines)):
            this_line = lines[mm].strip()
            testing_dir.append(this_line)


        subfolders_LQ = util.glob_file_list(self.LQ_root)
        subfolders_GT = util.glob_file_list(self.GT_root)
        if self.opt['phase'] == 'train':
            for subfolder_LQ, subfolder_GT in zip(subfolders_LQ, subfolders_GT):
                subfolder_name = osp.basename(subfolder_LQ)

                if (subfolder_name in testing_dir):
                    continue

                img_paths_LQ = util.glob_file_list(subfolder_LQ)
                img_paths_GT_all = util.glob_file_list(subfolder_GT)
                img_paths_GT = []
                for mm in range(len(img_paths_GT_all)):
                    if '.ARW' in img_paths_GT_all[mm] or 'half' in img_paths_GT_all[mm]:
                        continue
                    img_paths_GT.append(img_paths_GT_all[mm])

                max_idx = len(img_paths_LQ)
                self.data_info['path_LQ'].extend(img_paths_LQ)  # list of path str of images
                self.data_info['path_GT'].extend(img_paths_GT)
                self.data_info['folder'].extend([subfolder_name] * max_idx)
                for i in range(max_idx):
                    self.data_info['idx'].append('{}/{}'.format(i, max_idx))

                border_l = [0] * max_idx
                for i in range(self.half_N_frames):
                    border_l[i] = 1
                    border_l[max_idx - i - 1] = 1
                self.data_info['border'].extend(border_l)

                if self.cache_data:
                    self.imgs_LQ[subfolder_name] = img_paths_LQ
                    self.imgs_GT[subfolder_name] = img_paths_GT
        else:
            for subfolder_LQ, subfolder_GT in zip(subfolders_LQ, subfolders_GT):
                # for frames in each video:
                subfolder_name = osp.basename(subfolder_LQ)

                if not (subfolder_name in testing_dir):
                    continue

                img_paths_LQ = util.glob_file_list(subfolder_LQ)
                img_paths_GT_all = util.glob_file_list(subfolder_GT)
                img_paths_GT = []
                for mm in range(len(img_paths_GT_all)):
                    if '.ARW' in img_paths_GT_all[mm] or 'half' in img_paths_GT_all[mm]:
                        continue
                    img_paths_GT.append(img_paths_GT_all[mm])

                img_paths_LQ = img_paths_LQ[0:30]

                max_idx = len(img_paths_LQ)
                self.data_info['path_LQ'].extend(img_paths_LQ)  # list of path str of images
                self.data_info['path_GT'].extend(img_paths_GT)
                self.data_info['folder'].extend([subfolder_name] * max_idx)
                for i in range(max_idx):
                    self.data_info['idx'].append('{}/{}'.format(i, max_idx))

                border_l = [0] * max_idx
                for i in range(self.half_N_frames):
                    border_l[i] = 1
                    border_l[max_idx - i - 1] = 1
                self.data_info['border'].extend(border_l)

                if self.cache_data:
                    self.imgs_LQ[subfolder_name] = img_paths_LQ
                    self.imgs_GT[subfolder_name] = img_paths_GT



    def __getitem__(self, index):

        if self.opt['phase'] == 'train':
            folder = self.data_info['folder'][index]
            idx, max_idx = self.data_info['idx'][index].split('/')
            idx, max_idx = int(idx), int(max_idx)
            border = self.data_info['border'][index]

            img_LQ_path = self.imgs_LQ[folder][idx:idx + 1]
            img_GT_path = self.imgs_GT[folder][0]
            img_GT_path = [img_GT_path]

            img_LQ = util.read_img_seq2(img_LQ_path, self.opt['train_size'])
            img_LQ = img_LQ[0]
            img_GT = util.read_img_seq2(img_GT_path, self.opt['train_size'])
            img_GT = img_GT[0]

            LQ_size = self.opt['LQ_size']
            GT_size = self.opt['GT_size']

            _, H, W = img_GT.shape  # real img size

            rnd_h = random.randint(0, max(0, H - GT_size))
            rnd_w = random.randint(0, max(0, W - GT_size))
            img_LQ = img_LQ[:, rnd_h:rnd_h + GT_size, rnd_w:rnd_w + GT_size]
            img_GT = img_GT[:, rnd_h:rnd_h + GT_size, rnd_w:rnd_w + GT_size]

            img_LQ_l = [img_LQ]
            img_LQ_l.append(img_GT)
            rlt = util.augment_torch(img_LQ_l, self.opt['use_flip'], self.opt['use_rot'])
            img_LQ = rlt[0]
            img_GT = rlt[1]
            _, newH, newW = img_LQ.shape
            img_LQ2 = np.transpose(img_LQ.numpy(), (1, 2, 0))
            img_LQ2 = cv2.resize(img_LQ2, (newW // 2, newH // 2))
            img_LQ4 = cv2.resize(img_LQ2, (newW // 4, newH // 4))
            img_LQ2 = torch.tensor(np.transpose(img_LQ2, (2, 0, 1)), dtype=torch.float32)
            img_LQ4 = torch.tensor(np.transpose(img_LQ4, (2, 0, 1)), dtype=torch.float32)

            return {
                'LQs': img_LQ,
                'LQs2': img_LQ2,
                'LQs4': img_LQ4,
                'GT': img_GT,
                'folder': folder,
                'idx': self.data_info['idx'][index],
                'border': 0
            }
        else:
            folder = self.data_info['folder'][index]
            idx, max_idx = self.data_info['idx'][index].split('/')
            idx, max_idx = int(idx), int(max_idx)
            border = self.data_info['border'][index]

            select_idx = util.index_generation(idx, max_idx, self.opt['N_frames'], padding=self.opt['padding'])
            imgs_LQ_path = []
            for mm in range(len(select_idx)):
                imgs_LQ_path.append(self.imgs_LQ[folder][select_idx[mm]])
            img_GT_path = self.imgs_GT[folder][0]
            img_GT_path = [img_GT_path]

            imgs_LQ = util.read_img_seq2(imgs_LQ_path, self.opt['train_size'])
            img_GT = util.read_img_seq2(img_GT_path, self.opt['train_size'])
            img_GT = img_GT[0]

            img_LQ_l = list(imgs_LQ.unbind(0))
            _, newH, newW = img_LQ_l[2].shape
            img_LQ2 = np.transpose(img_LQ_l[2].numpy(), (1, 2, 0))
            img_LQ2 = cv2.resize(img_LQ2, (newW // 2, newH // 2))
            img_LQ4 = cv2.resize(img_LQ2, (newW // 4, newH // 4))
            img_LQ2 = torch.tensor(np.transpose(img_LQ2, (2, 0, 1)), dtype=torch.float32)
            img_LQ4 = torch.tensor(np.transpose(img_LQ4, (2, 0, 1)), dtype=torch.float32)

            return {
                'LQs': img_LQ_l[2],
                'LQs2': img_LQ2,
                'LQs4': img_LQ4,
                'GT': img_GT,
                'folder': folder,
                'idx': self.data_info['idx'][index],
                'border': 0
            }




    def __len__(self):
        return len(self.data_info['path_LQ'])
