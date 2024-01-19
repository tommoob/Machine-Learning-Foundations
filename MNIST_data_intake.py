import os
from matplotlib import pyplot as plot
import numpy as np


class MNISTDataInjest:
    def __init__(self, images, labels):
        self.images = images
        self.labels = labels

    def injest_labels(self, path_dir, byte_num, dir):
        count = 0
        f = open(os.path.join(path_dir, dir), mode="rb")
        fileContent = f.read(byte_num)
        how_many_bytes = 10007
        if "train" in dir:
            how_many_bytes = 60007
        res_arr = np.zeros((how_many_bytes))
        while f and count < how_many_bytes:
            fileContent = f.read(byte_num)
            conv_data = int.from_bytes(fileContent, byteorder='big')
            res_arr[count] = conv_data
            count += 1
        # print(type(fileContents), fileContent)
        self.labels[dir] = res_arr[7:]
        print(self.labels[dir])
        print(len(self.labels[dir]))
        f.close()


    def injest_images(self, path_dir, dir):
        bin_im = np.fromfile(os.path.join(path_dir, dir), dtype='uint8')[16:]
        num_imgs = (bin_im.shape[0]) / (28 * 28)
        imgs = bin_im.reshape(round(num_imgs), 28, 28)
        num_row, num_col = 4, 4
        num = num_row * num_col
        fig, axes = plot.subplots(num_row, num_col, figsize=(1.5 * num_col, 2 * num_row))
        for i in range(num):
            ax = axes[i // num_col, i % num_col]
            ax.imshow(imgs[i], cmap='gray')
            ax.set_title(i)
        plot.tight_layout()
        #plot.show()
        self.images[dir] = imgs
        return 
        

    def select_for_integer(self, lab_arr, target_int):
        
        lab_arr = np.where(lab_arr == float(target_int), 1, 0)
        return lab_arr
