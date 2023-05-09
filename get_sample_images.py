import torch
import numpy as np
from torchvision import models, transforms
from PIL import Image



shuffle = False
batch_size = 1

#class Image_Dataset(torch.utils.data.Dataset):
#
#        def __init__(self, dataset, labels, transforms):
#                self.data = np.transpose(dataset, (0, 3, 1, 2))
#                self.labels = labels
#                self.transforms = transforms
#
#        def __len__(self):
#                return len(self.data)
#
#        def __getitem__(self, idx):
#                return self.transforms(self.data[idx].float()), self.labels[idx]


class Image_Dataset(torch.utils.data.Dataset):
	
        def __init__(self, dataset, labels):
                self.data = np.transpose(dataset, (0, 3, 1, 2))
                self.labels = labels

        def __len__(self):
                return len(self.data)

        def __getitem__(self, idx):
                return self.data[idx].float(), self.labels[idx]



train_data = torch.load("dataset.pt")
train_labels = torch.load("labels.pt")
val_data = torch.load("validation_dataset.pt")
val_labels = torch.load("validation_labels.pt")


normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229,0.224,0.225])

train_dataset = Image_Dataset(train_data, train_labels)     #, normalize)
val_dataset = Image_Dataset(val_data, val_labels)     #, normalize)

train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle)
val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=shuffle)

iter_train = iter(train_dataloader)

for i in range(0, 10):
	data = next(iter_train)

	image, labels = data[0], data[1]
	image = (image - image.min()) / (image.max() - image.min())

	img = transforms.ToPILImage()(image[0])

	img.save("/home/thorkelsdottigl/NIH2022/sample_images/sample_image_" + str(i) + ".png")
