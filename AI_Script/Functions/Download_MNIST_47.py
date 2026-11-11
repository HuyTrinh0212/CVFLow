import torch
from torchvision import datasets, transforms
import os
from PIL import Image

# Define label mapping for bymerge (47 classes)
# labels_list = [
#     "0", "1", "2", "3", "4", "5", "6", "7", "8", "9",
#     "A", "B", "C", "D", "E", "F", "G", "H", "I", "J",
#     "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T",
#     "U", "V", "W", "X", "Y", "Z",
#     "a", "b", "d", "e", "f", "g", "h", "n", "q", "r", "t"
# ]
labels_list = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
 '10', '11', '12', '13', '14', '15', '16', '17', '18', '19',
 '20', '21', '22', '23', '24', '25', '26', '27', '28', '29',
 '30', '31', '32', '33', '34', '35', '36', '37', '38', '39',
 '40', '41', '42', '43', '44', '45', '46']


# Load EMNIST bymerge dataset with correction transform
transform = transforms.Compose([
    transforms.ToTensor(),
    lambda img: torch.rot90(img, k=3, dims=[1, 2]),
    lambda img: transforms.functional.hflip(img)
])

train_dataset = datasets.EMNIST(root='./data', split='bymerge', train=True, download=True, transform=transform)

# Create directories
os.makedirs('/home/bht/AI/data/Dataset_MNIST_47_1000/images', exist_ok=True)
os.makedirs('/home/bht/AI/data/Dataset_MNIST_47_1000/labels', exist_ok=True)

# Save first 10000 images and labels
for i, (img, label) in enumerate(train_dataset):
    if i >= 1000:
        break
    img = transforms.ToPILImage()(img)
    img.save(f'/home/bht/AI/data/Dataset_MNIST_47_1000/images/number_{i}.jpg')
    with open(f'/home/bht/AI/data/Dataset_MNIST_47_1000/labels/number_{i}.txt', 'w') as f:
        f.write(labels_list[label])