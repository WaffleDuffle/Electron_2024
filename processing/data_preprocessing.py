import torch
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

# print(f"PyTorch version: {torch.__version__}\ntorchvision version: {torchvision.__version__}")

from torchvision.datasets import ImageFolder
from torchvision.transforms import ToTensor

import os
print(os.listdir("C:\\Users\\denis\\Desktop\\ELECTRON\\hackathon\\resurse"))

transform = transforms.Compose([transforms.Resize((32, 32)), transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])

### Train Sets ###

trainset = datasets.ImageFolder("C:\\Users\\denis\\Desktop\\ELECTRON\\hackathon\\resurse\\train_set\\train_folder", transform=transform)
print(len(trainset))
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=False, drop_last = True)

# print(len(trainset), len(trainloader))

trainnoisyset = datasets.ImageFolder("C:\\Users\\denis\\Desktop\\ELECTRON\\hackathon\\resurse\\train_set\\train_noisy_folder", transform=transform)
trainnoisyloader = torch.utils.data.DataLoader(trainnoisyset, batch_size=64, shuffle=False, drop_last = True)

#######################################################

### Val sets ###

valset = datasets.ImageFolder("C:\\Users\\denis\\Desktop\\ELECTRON\\hackathon\\resurse\\val_set\\val_folder", transform=transform)
valloader = torch.utils.data.DataLoader(valset, batch_size = 128, shuffle = False)

valnoisyset = datasets.ImageFolder("C:\\Users\\denis\\Desktop\\ELECTRON\\hackathon\\resurse\\val_set\\val_noisy_folder", transform=transform)
valloader = torch.utils.data.DataLoader(valnoisyset, batch_size = 128, shuffle = False)

#######################################################

# for images, labels in trainloader:

#     denormalized_images = (images * 0.5) + 0.5
#     numpy_images = denormalized_images.numpy()
#     plt.figure(figsize=(10, 10))
#     for i in range(len(images)):
#         plt.subplot(4, 8, i+1)
#         plt.imshow(numpy_images[i].transpose((1, 2, 0)))  
#         plt.axis('off')
#         class_name = trainset.classes[labels[i]]
#         plt.title(f"{class_name}")
#     plt.show()
#     break 

# for images, labels in trainnoisyloader:

#     denormalized_images = (images * 0.5) + 0.5
#     numpy_images = denormalized_images.numpy()
#     plt.figure(figsize=(10, 10))
#     for i in range(len(images)):
#         plt.subplot(4, 8, i+1)
#         plt.imshow(numpy_images[i].transpose((1, 2, 0)))  
#         plt.axis('off')
#         class_name = trainnoisyset.classes[labels[i]]
#         plt.title(f"{class_name}")
#     plt.show()
#     break  



