import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import numpy as np
import pyvips
from PIL import Image
import cv2
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



#image = vips2numpy(pyvips.Image.new_from_file("/data/Jiang_Lab/datashare/Beibei/ST/Gudrun/He_2020_Breast.Cancer/BC23209_C1/HE.jpg"))
image = vips2numpy(pyvips.Image.new_from_file("/home/thorkelsdottigl/NIH2022/data_samples/2023/398A.tif"))


spot_coords = np.zeros((10,2))
#spot_coords[0][0] = 5207
#spot_coords[0][1] = 1161
#spot_coords[1][0] = 5513
#spot_coords[1][1] = 1160
#spot_coords[2][0] = 4907
#spot_coords[2][1] = 1448
#spot_coords[3][0] = 5210
#spot_coords[3][1] = 1448
#spot_coords[4][0] = 5517
#spot_coords[4][1] = 1458
#spot_coords[5][0] = 4618
#spot_coords[5][1] = 1461
#spot_coords[6][0] = 5204
#spot_coords[6][1] = 1740
#spot_coords[7][0] = 4917
#spot_coords[7][1] = 1745
#spot_coords[8][0] = 4617
#spot_coords[8][1] = 1741
#spot_coords[9][0] = 5528
#spot_coords[9][1] = 1744

spot_coords[0][0] = 13735
spot_coords[0][1] = 3734
spot_coords[1][0] = 16743
spot_coords[1][1] = 5336
spot_coords[2][0] = 14666
spot_coords[2][1] = 6949
spot_coords[3][0] = 5769
spot_coords[3][1] = 7170
spot_coords[4][0] = 5307
spot_coords[4][1] = 7171
spot_coords[5][0] = 5076
spot_coords[5][1] = 7171
spot_coords[6][0] = 4845
spot_coords[6][1] = 7172
spot_coords[7][0] = 4613
spot_coords[7][1] = 7172
spot_coords[8][0] = 4382
spot_coords[8][1] = 7173
spot_coords[9][0] = 4151
spot_coords[9][1] = 7174

spot_coords = spot_coords[:, (1, 0)]


#spot_ids = ["19x5", "20x5", "18x6", "19x6", "20x6", "17x6", "19x7", "18x7", "17x7", "20x7"]
spot_ids = ["6x32", "14x6", "22x24", "23x101", "23x105", "23x107", "23x109", "23x111", "23x113", "23x115"]


for i in range(0, 10):
	cv2.circle(img=image, center = (int(spot_coords[i][0]),int(spot_coords[i][1])), radius = 122, color =(255,0,0), thickness=10)
	cv2.putText(img=image, text = spot_ids[i], fontFace =  1, fontScale=3, color = (255, 0, 0), thickness = 3, org = (int(spot_coords[i][0])+40 ,int(spot_coords[i][1])+150))


img = transforms.ToPILImage()(image)
#img.save("/home/thorkelsdottigl/NIH2022/sample_images/BC23209_C1_circles.jpg")
img.save("/home/thorkelsdottigl/NIH2022/sample_images/398A_circles.tif")
