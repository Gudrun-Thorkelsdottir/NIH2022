import os
import csv
from PIL import Image
import numpy as np

vipshome = 'c:\\vips-dev-8.13.3\\bin'
os.environ['PATH'] = vipshome + ';' + os.environ['PATH']


import pyvips
import sys
import torch
import random
from torchvision import transforms

def vips2numpy(vi):
    format_to_dtype = {
        'uchar': np.uint8,
        'char': np.int8,
        'ushort': np.uint16,
        'short': np.int16,
        'uint': np.uint32,
        'int': np.int32,
        'float': np.float32,
        'double': np.float64,
        'complex': np.complex64,
        'dpcomplex': np.complex128,
    }
    return np.ndarray(buffer=vi.write_to_memory(),
                      dtype=format_to_dtype[vi.format],
                      shape=[vi.height, vi.width, vi.bands])


def extract_data(f1, f2, f3, split_data, shuffle, visium):

        dataset = []
        labels = []
        spot_coords = []
        tile_size = 224
        half_tile = int(tile_size/2)

        #add data from all images given
        for i in range(len(f1)):

                #load data into memory
                image = vips2numpy(pyvips.Image.new_from_file(f1[i]))
                if visium:
                        tissue_pos = np.loadtxt(f2[i], delimiter=',', usecols=range(2,6), dtype=np.uint16)
                else:
                        with open(f2[i]) as f:
                                tissue_pos = np.loadtxt((line.replace('x', ',') for line in f), delimiter=',', skiprows=1)
                with open(f3[i]) as f:
                        num_cols = len(f.readline().split(","))
                propMat = np.genfromtxt(f3[i], delimiter=',', names=True, usecols=range(1, num_cols))
                size = image.shape

                #for each spot, add to dataset if it exists in both f3 and f4
                for row in tissue_pos:

                        spot_id = (str(int(row[0])), str(int(row[1])))
                        try:
                                labels.append(torch.tensor(propMat[(spot_id[0] + 'x' + spot_id[1])][:13]))
                                pixel_pos = (int(row[2]), int(row[3]))
                                t = image[max(0, pixel_pos[0] - half_tile):min(size[0], pixel_pos[0] + half_tile),\
                                max(0, pixel_pos[1] - half_tile):min(size[1], pixel_pos[1] + half_tile)]
                                if t.shape == (tile_size, tile_size, 3):
                                        dataset.append(torch.from_numpy(t))
                                spot_coords.append(spot_id)
                        except ValueError:
                                pass



        #split into train and val data if split_data is True
        if split_data==False:
                return [torch.stack(dataset, 0), torch.stack(labels, 0)]
        else:

                #randomize train and val datasets if shuffle is True
                if shuffle:
                        train_indices = random.sample(range(0, len(dataset)), int(len(dataset) * 0.8))
                        val_indices = np.delete(range(0, len(dataset)), train_indices)
                        train_data = [dataset[i] for i in train_indices]
                        train_labels = [labels[i] for i in train_indices]
                        val_data = [dataset[i] for i in val_indices]
                        val_labels = [labels[i] for i in val_indices]
                else:
                        train_data = dataset[:int(len(dataset) * 0.8)]
                        train_labels = labels[:int(len(labels) * 0.8)]
                        val_data = dataset[int(len(dataset) * 0.8):]
                        val_labels = labels[int(len(labels) * 0.8):]


                return [torch.stack(train_data, 0), torch.stack(train_labels, 0),\
                         torch.stack(val_data, 0), torch.stack(val_labels, 0)]

if __name__ == '__main__':

        visium = False

        if visium:
                '''
                f1 = ["/data/Jiang_Lab/datashare/Beibei/ST/10x_SpatialDatasets_Breast.Cancer/Version1.0.0_Breast.Cancer_rep1/V1_Breast_Cancer_Block_A_Section_1_image.tif", "/data/Jiang_Lab/datashare/Beibei/ST/10x_SpatialDatasets_Breast.Cancer/Version1.0.0_Breast.Cancer_rep2/V1_Breast_Cancer_Block_A_Section_2_image.tif"] #, "/data/Jiang_Lab/datashare/ST/10x_SpatialDatasets_Breast.Cancer/Version1.2.0_Breast.Cancer.Invasive.Lobular.Carcinoma/Targeted_Visium_Human_BreastCancer_Immunology_image.tif", "/data/Jiang_Lab/datashare/ST/10x_SpatialDatasets_Breast.Cancer/Version1.3.0_Breast.Cancer/Visium_FFPE_Human_Breast_Cancer_image.tif"]

                f2 = ["/data/Jiang_Lab/datashare/Beibei/ST/10x_SpatialDatasets_Breast.Cancer/Version1.0.0_Breast.Cancer_rep1/spatial/tissue_positions_list.csv", "/data/Jiang_Lab/datashare/Beibei/ST/10x_SpatialDatasets_Breast.Cancer/Version1.0.0_Breast.Cancer_rep2/spatial/tissue_positions_list.csv"] #, "/data/Jiang_Lab/datashare/ST/10x_SpatialDatasets_Breast.Cancer/Version1.2.0_Breast.Cancer.Invasive.Lobular.Carcinoma/tissue_positions_list.csv", "/data/Jiang_Lab/datashare/ST/10x_SpatialDatasets_Breast.Cancer/Version1.3.0_Breast.Cancer/tissue_positions_list.csv"]

                f3 = ["/data/Jiang_Lab/datashare/Beibei/ST/10x_SpatialDatasets_Breast.Cancer/Version1.0.0_Breast.Cancer_rep1/propMat_SpaCE.csv", "/data/Jiang_Lab/datashare/Beibei/ST/10x_SpatialDatasets_Breast.Cancer/Version1.0.0_Breast.Cancer_rep2/propMat_SpaCE.csv"] #, "/data/Jiang_Lab/datashare/ST/10x_SpatialDatasets_Breast.Cancer/Version1.2.0_Breast.Cancer.Invasive.Lobular.Carcinoma/propMat_SpaCE.csv", "/data/Jiang_Lab/datashare/ST/10x_SpatialDatasets_Breast.Cancer/Version1.3.0_Breast.Cancer/propMat_SpaCE.csv"]
		'''

                f1 = ["/home/thorkelsdottigl/NIH2022/data_samples/2023/398A.tif"]
                f2 = ["/home/thorkelsdottigl/NIH2022/data_samples/2023/GSM6433624_398A_tissue_positions_list.csv"]
                f3 = ["/home/thorkelsdottigl/NIH2022/data_samples/2023/propMat_SpaCE.csv"]




                #if >=5 images, hold one for val data
                if len(f1) > 1:
                        [dataset, labels] = extract_data(f1[:-1], f2[:-1], f3[:-1], False, False, visium)
                        [val_dataset, val_labels] = extract_data([f1[len(f1)-1]], [f2[len(f2)-1]], [f3[len(f3)-1]], False, False, visium)
                else:
                        [dataset, labels, val_dataset, val_labels] = extract_data(f1, f2, f3, True, False, visium)


        else:
                f1 = []
                f2 = []
                f3 = []
                #folders = os.listdir('/data/Jiang_Lab/datashare/ST/He_2020_Breast.Cancer')
                folders = os.listdir('/data/Jiang_Lab/datashare/Beibei/ST/Gudrun/He_2020_Breast.Cancer/')

                #for folder in folders:
                for i in range(0, 10):
                        f1.append('/data/Jiang_Lab/datashare/Beibei/ST/Gudrun/He_2020_Breast.Cancer/' + folders[i] + '/HE.jpg')
                        f2.append('/data/Jiang_Lab/datashare/Beibei/ST/Gudrun/He_2020_Breast.Cancer/' + folders[i] + '/spot_coordinates.csv')
                        f3.append('/data/Jiang_Lab/datashare/Beibei/ST/Gudrun/He_2020_Breast.Cancer/' + folders[i] + '/propMat_SpaCE.csv')

                split = int(len(f1) * 0.8)
                [dataset, labels] = extract_data(f1[:split], f2[:split], f3[:split], False, False, visium)
                [val_dataset, val_labels] = extract_data(f1[split:], f2[split:], f3[split:], False, False, visium)
                #[dataset, labels, val_dataset, val_labels] = extract_data(f1, f2, f3, True, False, visium)

        torch.save(dataset, 'dataset.pt')
        torch.save(labels, 'labels.pt')
        torch.save(val_dataset, 'validation_dataset.pt')
        torch.save(val_labels, 'validation_labels.pt')

