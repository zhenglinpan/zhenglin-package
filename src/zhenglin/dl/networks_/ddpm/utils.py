import os
from typing import Any
import torch
import torchvision
from PIL import Image
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader


def plot_images(images):
    plt.figure(figsize=(32, 32))
    plt.imshow(torch.cat([
        torch.cat([i for i in images.cpu()], dim=-1),
    ], dim=-2).permute(1, 2, 0).cpu())
    plt.show()


def save_images(images, path, **kwargs):
    grid = torchvision.utils.make_grid(images, **kwargs)
    ndarr = grid.permute(1, 2, 0).to('cpu').numpy()
    im = Image.fromarray(ndarr)
    im.save(path)


def get_data(args):
    transforms = torchvision.transforms.Compose([
        torchvision.transforms.Resize(80),  # args.image_size + 1/4 *args.image_size
        torchvision.transforms.RandomResizedCrop(args.image_size, scale=(0.8, 1.0)),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    dataset = torchvision.datasets.ImageFolder(args.dataset_path, transform=transforms)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    return dataloader


def setup_logging(run_name):
    os.makedirs("models", exist_ok=True)
    os.makedirs("results", exist_ok=True)
    os.makedirs(os.path.join("models", run_name), exist_ok=True)
    os.makedirs(os.path.join("results", run_name), exist_ok=True)


import random
import time
import datetime
import sys

from torch.autograd import Variable
import torch
from visdom import Visdom
import numpy as np
import cv2
from matplotlib import pyplot as plt
import os
from tqdm import tqdm

def tensor2image(tensor):
    image = 127.5*(tensor[0].cpu().float().numpy() + 1.0)
    if image.shape[0] == 1:
        image = np.tile(image, (3,1,1))
    return image.astype(np.uint8)

class Logger():
    def __init__(self, n_epochs, batches_epoch):
        self.viz = Visdom()
        self.n_epochs = n_epochs
        self.batches_epoch = batches_epoch
        self.epoch = 1
        self.batch = 1
        self.prev_time = time.time()
        self.mean_period = 0
        self.losses = {}
        self.loss_windows = {}
        self.image_windows = {}


    def log(self, losses=None, images=None):
        self.mean_period += (time.time() - self.prev_time)
        self.prev_time = time.time()

        sys.stdout.write('\rEpoch %03d/%03d [%04d/%04d] -- ' % (self.epoch, self.n_epochs, self.batch, self.batches_epoch))

        for i, loss_name in enumerate(losses.keys()):
            if loss_name not in self.losses:
                self.losses[loss_name] = losses[loss_name].data[0]
            else:
                self.losses[loss_name] += losses[loss_name].data[0]

            if (i+1) == len(losses.keys()):
                sys.stdout.write('%s: %.4f -- ' % (loss_name, self.losses[loss_name]/self.batch))
            else:
                sys.stdout.write('%s: %.4f | ' % (loss_name, self.losses[loss_name]/self.batch))

        batches_done = self.batches_epoch*(self.epoch - 1) + self.batch
        batches_left = self.batches_epoch*(self.n_epochs - self.epoch) + self.batches_epoch - self.batch 
        sys.stdout.write('ETA: %s' % (datetime.timedelta(seconds=batches_left*self.mean_period/batches_done)))

        # Draw images
        for image_name, tensor in images.items():
            if image_name not in self.image_windows:
                self.image_windows[image_name] = self.viz.image(tensor2image(tensor.data), opts={'title':image_name})
            else:
                self.viz.image(tensor2image(tensor.data), win=self.image_windows[image_name], opts={'title':image_name})

        # End of epoch
        if (self.batch % self.batches_epoch) == 0:
            # Plot losses
            for loss_name, loss in self.losses.items():
                if loss_name not in self.loss_windows:
                    self.loss_windows[loss_name] = self.viz.line(X=np.array([self.epoch]), Y=np.array([loss/self.batch]), 
                                                                    opts={'xlabel': 'epochs', 'ylabel': loss_name, 'title': loss_name})
                else:
                    self.viz.line(X=np.array([self.epoch]), Y=np.array([loss/self.batch]), win=self.loss_windows[loss_name], update='append')
                # Reset losses for next epoch
                self.losses[loss_name] = 0.0

            self.epoch += 1
            self.batch = 1
            sys.stdout.write('\n')
        else:
            self.batch += 1

        

class ReplayBuffer():
    def __init__(self, max_size=50):
        assert (max_size > 0), 'Empty buffer or trying to create a black hole. Be careful.'
        self.max_size = max_size
        self.data = []

    def push_and_pop(self, data):
        to_return = []
        for element in data.data:
            element = torch.unsqueeze(element, 0)
            if len(self.data) < self.max_size:
                self.data.append(element)
                to_return.append(element)
            else:
                if random.uniform(0,1) > 0.5:
                    i = random.randint(0, self.max_size-1)
                    to_return.append(self.data[i].clone())
                    self.data[i] = element
                else:
                    to_return.append(element)
        return Variable(torch.cat(to_return))

class LambdaLR():
    def __init__(self, n_epochs, offset, decay_start_epoch):
        assert ((n_epochs - decay_start_epoch) > 0), "Decay must start before the training session ends!"
        self.n_epochs = n_epochs
        self.offset = offset
        self.decay_start_epoch = decay_start_epoch

    def step(self, epoch):
        return 1.0 - max(0, epoch + self.offset - self.decay_start_epoch)/(self.n_epochs - self.decay_start_epoch)

def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        torch.nn.init.normal(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm2d') != -1:
        torch.nn.init.normal(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant(m.bias.data, 0.0)

def save_checkpoint(model, epoch, args):
    model_name = str(args.dataset)+'_'+'cycleGAN'+'_'+args.version
    model_out_path_folder = os.path.join(args.model_dir,'model',model_name)
    model_out_path = os.path.join(model_out_path_folder,"model_epoch_{}.pth".format(epoch))
    state = {"epoch": epoch ,"model": model}
    if not os.path.exists(model_out_path_folder):
        os.makedirs(model_out_path_folder)
    if epoch%args.step2save == 0:
        torch.save(state, model_out_path)
        print("Checkpoint saved to {}".format(model_out_path))


class easymatch:
    def __init__(self, reference, moving, match_list=None, case_dir=None):
        self.reference = reference
        self.moving = moving
        self.roi = [75, 75, 435, 435]
        self.match_list = match_list
        self.case_dir = case_dir

        if self.match_list is None:
            self.match_list = [0] * len(self.moving)
            for i in range(len(self.moving)):
                idx = round(i / (len(self.moving) / len(self.reference)))
                self.match_list[i] = min(idx, len(self.reference) - 1)
    
    def fast_match(self):
        if self.case_dir is None:
            raise Exception("External case_dir should be provided.")
        refs = list()
        movs = list()
        for mov_idx, ref_idx in enumerate(self.match_list):
            mov = self.moving[mov_idx]
            ref = self.reference[ref_idx]
            ref_mat, mov_mat = self.fetch_matries(self.case_dir)
            
            ref = cv2.warpAffine(ref, ref_mat, (ref.shape[1], ref.shape[0]), flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)
            mov = cv2.warpAffine(mov, mov_mat, (mov.shape[1], mov.shape[0]), flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)
            
            mov = mov[self.roi[0]:self.roi[2], self.roi[1]:self.roi[3]]
            mov = cv2.resize(mov, (256, 256), interpolation=cv2.INTER_CUBIC)
            ref = cv2.resize(ref, (256, 256), interpolation=cv2.INTER_CUBIC)
            refs.append(ref)
            movs.append(mov)
        
        return refs, movs, self.match_list
    
    def easy_match(self):
        refs = list()
        movs = list()
        for mov_idx, ref_idx in tqdm(enumerate(self.match_list), ascii=' >='):
            mov = self.moving[mov_idx]
            ref = self.reference[ref_idx]
            mov = mov[self.roi[0]:self.roi[2], self.roi[1]:self.roi[3]]
            mov = cv2.resize(mov, (256, 256), interpolation=cv2.INTER_CUBIC)
            ref = cv2.resize(ref, (256, 256), interpolation=cv2.INTER_CUBIC)
            refs.append(ref)
            movs.append(mov)
        
        return refs, movs, self.match_list

    def fetch_matries(self,case_dir):
        matries_root = '/model/zhenglin/ct/matries'
        case_name = case_dir.split('/')[-1]
        matries_dir = os.path.join(matries_root, case_name)
        clean_mat = np.load(matries_dir + '/cleans.npy')
        metal_mat = np.load(matries_dir + '/metals.npy')
        
        return clean_mat, metal_mat
    
class match2:
    def __init__(self, reference, moving, match_list=None):
        self.reference = reference
        self.moving = moving
        self.roi = [75, 75, 435, 435]
        self.ror = [0, 0, 512, 512]
        self.match_list = None
        self.avg_centroid = True
        self.target_ratio = 0.03
        
    def do_match2(self):
        # determine the matching by simply length ratio
        if self.match_list is None:
            self.match_list = [0] * len(self.moving)
            for i in range(len(self.moving)):
                idx = round(i / (len(self.moving) / len(self.reference)))
                self.match_list[i] = min(idx, len(self.reference) - 1)
        
        avg_wrp_f, avg_wrp_m = self.get_transform_matrix()
        
        # doing registration for all
        refs = list()
        movs = list()
        for mov_idx, ref_idx in enumerate(self.match_list):
            mov = self.moving[mov_idx]
            ref = self.reference[ref_idx]
            
            ref = self.__register_opencv(fix=None, mov=ref, warp_matrix=avg_wrp_f)
            mov = self.__register_opencv(fix=None, mov=mov, warp_matrix=avg_wrp_m)
            
            wrp = self.__register_opencv(ref, mov)
            ref = ref[self.roi[0]:self.roi[2], self.roi[1]:self.roi[3]]
            wrp = wrp[self.roi[0]:self.roi[2], self.roi[1]:self.roi[3]]
            refs.append(ref)
            movs.append(wrp)
        
        return refs, movs, self.match_list

    def get_transform_matrix(self,):
        # average centroid
        avg_wrp_m = np.eye(2, 3, dtype=np.float32)
        avg_ratio = 0
        for mov_idx in range(len(self.moving)):
            mov = self.moving[mov_idx]
            mov = mov[self.roi[0]:self.roi[2], self.roi[1]:self.roi[3]]
            mov = cv2.resize(mov, (512, 512), interpolation=cv2.INTER_CUBIC)
            mov = self.scale255(mov)
            wrp_mtx, ratio = self.find_centroid(mov, self.ror, 'both')
            avg_wrp_m += wrp_mtx / len(self.moving)
            avg_ratio += ratio / len(self.moving)
            # if mov_idx == 0: 
            #     plt.imsave('./results/mov.jpg', mov, cmap='gray')
        scaling = avg_ratio / self.target_ratio
        avg_wrp_m[0, 0], avg_wrp_m[1, 1] = scaling, scaling
        avg_wrp_m[0, 2] += (1 - scaling) * mov.shape[1] * 0.5
        avg_wrp_m[1, 2] += (1 - scaling) * mov.shape[1] * 0.5
        # print(avg_wrp_m)
        
        avg_wrp_f = np.eye(2, 3, dtype=np.float32)
        avg_ratio = 0
        for ref_idx in range(len(self.reference)):
            ref = self.reference[ref_idx]
            ref = self.scale255(ref)
            wrp_mtx, ratio = self.find_centroid(ref, self.ror, 'both')
            avg_wrp_f += wrp_mtx / len(self.reference)
            avg_ratio += ratio / len(self.reference)
            # if ref_idx == 0:
                # plt.imsave('./results/ref.jpg', ref, cmap='gray')
        scaling = avg_ratio / self.target_ratio
        avg_wrp_f[0, 0], avg_wrp_f[1, 1] = scaling, scaling
        avg_wrp_f[0, 2] += (1 - scaling) * ref.shape[1] * 0.5
        avg_wrp_f[1, 2] += (1 - scaling) * ref.shape[1] * 0.5
        # print(avg_wrp_f)
        
        return avg_wrp_f, avg_wrp_m
        
    def get_largest_blob(self, mat):
        """
            Return: dst->masked mat
                    largest_ratio->largest blob area:A, image area:B, largest_ratio:A/B
        """
        contours, _= cv2.findContours(cv2.Canny(mat, 50, 200), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        largest_contour = max(contours, key=cv2.contourArea)
        largest_ratio = cv2.contourArea(largest_contour) / (mat.shape[0] * mat.shape[1])
        mask = np.zeros_like(mat)
        dst = cv2.drawContours(mask, [largest_contour], 0, (255), cv2.FILLED)
        return dst, largest_ratio

    def cal_shift(self, mat, direction='y'):
        assert direction.lower() in ['x', 'y', 'both']
        x_centroid_moving = np.mean(np.where(mat == 255)[1])
        y_centroid_moving = np.mean(np.where(mat == 255)[0])
        x_shift = int(x_centroid_moving - mat.shape[1] / 2)
        y_shift = int(y_centroid_moving - mat.shape[0] / 2)
        warp_matrix = np.eye(2, 3, dtype=np.float32)
        if direction == 'x':
            warp_matrix[0, 2] = x_shift
        elif direction == 'y':
            warp_matrix[1, 2] = y_shift
        else:
            warp_matrix[0, 2] = x_shift
            warp_matrix[1, 2] = y_shift
        
        return warp_matrix
    
    def scale255(self, mat):
        rand_name = './tmp/' + str(np.random.random()) + '.jpg'
        cv2.imwrite(rand_name, mat)
        converted = cv2.imread(rand_name, cv2.IMREAD_GRAYSCALE)
        os.remove(rand_name)
        return converted
    
    def find_centroid(self, mat, ror, direction='y'):
        # preprocess
        mat_ror = mat
        mat_ror = np.array(mat_ror).astype(np.uint8)
        # print(mat_ror.shape)
        # plt.imsave('./results/mat_ror.jpg', mat_ror, cmap='gray')
        _, mat_ror = cv2.threshold(mat_ror, 127, 255, cv2.THRESH_OTSU)
        # plt.imsave('./results/mat_rorthreshold.jpg', mat_ror, cmap='gray')
        kernel = np.ones((15, 15), np.uint8)
        mat_ror = cv2.morphologyEx(mat_ror, cv2.MORPH_CLOSE, kernel)
        mat_ror = cv2.morphologyEx(mat_ror, cv2.MORPH_OPEN, kernel)
        # plt.imsave('./results/mat_rormorphologyEx.jpg', mat_ror, cmap='gray')
        
        mat_ror, blob_ratio = self.get_largest_blob(mat_ror)
        # cal shift
        warp_matrix = self.cal_shift(mat_ror, direction)
        
        return warp_matrix, blob_ratio
    
    def __register_opencv(self, fix=None, mov=None, warp_matrix=None):
        h, w = mov.shape
        if warp_matrix is not None:
            return cv2.warpAffine(mov, warp_matrix, (w, h), flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)
        fix_crop = fix[self.ror[0]:self.ror[2], self.ror[1]:self.ror[3]]
        mov_crop = mov[self.ror[0]:self.ror[2], self.ror[1]:self.ror[3]]

        warp_mode = cv2.MOTION_TRANSLATION
        warp_matrix = np.eye(2, 3, dtype=np.float32)
        number_of_iterations = 5000
        termination_eps = 1e-10
        criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, number_of_iterations,  termination_eps)
        try:
            (_, warp_matrix) = cv2.findTransformECC(fix_crop,mov_crop,warp_matrix, warp_mode, criteria)
        except:
            return mov_crop
        
        warped = cv2.warpAffine(mov, warp_matrix, (w, h), flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)
        
        return warped

class MyPatch(object):
    def __call__(self, mat, patch_n):
        pass

def dissect(mat: torch.Tensor(), patch_n):
    n = int(np.sqrt(patch_n))
    assert n ** 2 == patch_n, "patch number must be a squred number."
    assert len(mat.shape) == 4, "input mat should have a dimension of 4"

    p_h, p_w = int(mat.shape[2]/n), int(mat.shape[3]/n)
    patches = list()
    for cnt in range(patch_n):
        i, j = cnt // n, cnt % n
        patch = mat[:, :, i*p_h:(i+1)*p_h, j*p_w:(j+1)*p_w]
        patches.append(patch)

    return patches

def stitch(patches: list):
    p = int(np.sqrt(len(patches)))
    n, c, h, w = patches[0].shape[:]
    large_patch = torch.zeros(n, c, h*p, w*p)
    for cnt in range(len(patches)):
        i, j = cnt // p, cnt % p
        large_patch[:, :, i*h:(i+1)*h, j*w:(j+1)*w] = patches[cnt]
        
    return large_patch
    