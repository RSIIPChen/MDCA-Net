import os
import h5py
import numpy as np
from tqdm import tqdm
from skimage.io import imread
import cv2

def gather_image_files(directory):
    file_list = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.lower().endswith(
                    ('.bmp', '.dib', '.png', '.jpg', '.jpeg', '.pbm', '.pgm', '.ppm', '.tif', '.tiff', '.npy')):
                file_list.append(os.path.join(root, file))
    return file_list

def convert_rgb_to_ycrcb(image_path):
    img = cv2.imread(image_path, 1)
    img = img.astype('float32')
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)  
    (B, G, R) = cv2.split(img)

    Y = 0.299 * R + 0.587 * G + 0.114 * B 
    Cr = (R - Y) * 0.713 + 0.5
    Cb = (B - Y) * 0.564 + 0.5

    return Y, cv2.merge([Cr, Cb])

def image_to_patches(image, window_size):
    channels = image.shape[0]
    height = image.shape[1]
    width = image.shape[2]
    total_patches = (width - window_size + 1) * (height - window_size + 1)
    patches = np.zeros([channels, window_size * window_size, total_patches], np.float32)

    index = 0
    for i in range(window_size):
        for j in range(window_size):
            
            patch = image[:, i:height - window_size + i + 1, j:width - window_size + j + 1]
            patches[:, index, :] = np.array(patch[:]).reshape(channels, total_patches)
            index += 1
    return patches.reshape([channels, window_size, window_size, total_patches])


data_name = "train"
img_size = 256  # patch size

ir_dir = "/data/sar/train"
vis_dir = "/data/opt/train"

ir_files = gather_image_files(ir_dir)
vis_files = gather_image_files(vis_dir)

assert len(ir_files) == len(vis_files), "IR and VIS directories must contain the same number of images."

h5f = h5py.File(os.path.join('/data', 'train_xingjiang1.h5'), 'w')
h5_ir = h5f.create_group('ir_patches')
h5_vis = h5f.create_group('vis_patches')

train_counter = 0
for IR_file, VIS_file in tqdm(zip(ir_files, vis_files), total=len(ir_files)):
    I_VIS, CBCR = convert_rgb_to_ycrcb(VIS_file)
    I_VIS = np.expand_dims(I_VIS, axis=0) / 255

    # 处理红外图像
    I_IR = imread(IR_file).astype(np.float32)

    if len(I_IR.shape) == 2:  
        I_IR = np.expand_dims(I_IR, -1) 
        I_IR = np.concatenate((I_IR, I_IR, I_IR), axis=-1)
    I_IR = I_IR.transpose(2, 0, 1) / 255.  

    I_IR_Patch_Group = image_to_patches(I_IR, img_size)
    I_VIS_Patch_Group = image_to_patches(I_VIS, img_size)

    for ii in range(I_IR_Patch_Group.shape[-1]):
        available_IR = I_IR_Patch_Group[0, :, :, ii][None, ...]
        available_VIS = I_VIS_Patch_Group[0, :, :, ii][None, ...]

        h5_ir.create_dataset(str(train_counter), data=available_IR, dtype=available_IR.dtype, shape=available_IR.shape)
        h5_vis.create_dataset(str(train_counter), data=available_VIS, dtype=available_VIS.dtype, shape=available_VIS.shape)
        train_counter += 1

h5f.close()

with h5py.File(os.path.join('data', "train.h5"), "r") as f:
    for key in f.keys():
        print(f[key], key, f[key].name)
