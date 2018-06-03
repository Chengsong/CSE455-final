import os
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset
from skimage import io, transform

class PaintingDataset(Dataset):
	def __init__(self, root, train=True, transform=None):
		self.root = root
		self.transform = transform

		if train:
			with open(os.path.join(root, 'train.csv')) as f:
				self.data = [(io.imread("./data/images/" + line.split(',')[0]), int(line.split(',')[1].strip()))
							 for line in f]

		else:
			with open(os.path.join(root, 'test.csv')) as f:
				self.data = [(io.imread("./data/images/" + line.split(',')[0]), int(line.split(',')[1].strip()))
							 for line in f]

	def __len__(self):
		return len(self.data)

	def __getitem__(self, idx):
		sample = self.data[idx]
		if self.transform:
			sample = (self.transform(sample[0]), sample[1])
		return sample


class PaintingDataLoader():
	def __init__(self, root, batchSize):
		transform = transforms.Compose(
		    [
			 transforms.ToPILImage(),
             transforms.RandomHorizontalFlip(),
		     transforms.ToTensor(),
		     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
		     ])

		transform_test = transforms.Compose([
		    transforms.ToTensor(),
		    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)), 
		])

		trainset = PaintingDataset(root=root, train=True, transform=transform)
		self.trainloader = torch.utils.data.DataLoader(trainset, batch_size=batchSize,
		                                          shuffle=True, num_workers=2)

		testset = PaintingDataset(root=root, train=False, transform=transform_test)
		self.testloader = torch.utils.data.DataLoader(testset, batch_size=batchSize,
		                                         shuffle=False, num_workers=2)

		self.classes = []
		with open(os.path.join(root, 'class-labels.txt')) as f:
			for line in f:
				name = line.split(',')[0]
				name = ' '.join([word.capitalize() for word in name.split('-')])
				self.classes.append(name)