import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

from dataset import MyDataset
from model import PSPNet

class PSPLoss(nn.Module):
	def __init__(self, aux_weight=0.4):
		super().__init__()
		self.aux_weight = aux_weight

	def forward(self, outputs, gts):
		"""
		Args
		outputs : output of PSPNet (tuple)
			(output=torch.Size([batch, n_classes, 475, 475]),
			output_aux=torch.Size([batch, n_classes, 475, 475]))

		gts : ground truth [batch, 475, 475]
		
		Returns
		loss : torch tensor
		"""
		loss = F.cross_entropy(outputs[0], gts, reduction='mean')
		loss_aux = F.cross_entropy(outputs[1], gts, reduction='mean')
		return loss + self.aux_weight * loss_aux

if __name__ == '__main__':
	loss_func = PSPLoss(aux_weight=0.4)
	img_dir = 'data/leftImg8bit/'
	anno_dir = 'data/gtFine/'
	phase ='val'
	batch = 2
	num_workers = 4
	model = PSPNet(n_classes = 35)
	dataset = MyDataset(img_dir = img_dir, anno_dir = anno_dir, phase = phase, h = 512, w = 1024)
	loader = DataLoader(dataset, batch_size = batch, 
				shuffle = True, pin_memory = True, num_workers = num_workers)
				
	device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
	model.to(device)
	for t, (img, anno_img) in enumerate(loader, 1):
		print(img.size())
		print(anno_img.size())
		img = img.to(device)
		anno_img = anno_img.to(device)
		outputs = model(img)
		print(outputs[0].size())
		print(outputs[1].size())
		loss = loss_func(outputs, anno_img.long())
		print(loss)
		print(loss.item())
		if t == 1:
			break
