import torch
from torch.utils.data import DataLoader
import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image

from model import PSPNet
from dataset import MyDataset
from config import test_parser
from visualize import unnormalize_show, PIL_show

if __name__ == '__main__':
	arg = test_parser()
	device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
	model = PSPNet(n_classes = arg.n_classes)
	if arg.path is not None:
		print('load model ...')
		model.load_state_dict(torch.load(arg.path, map_location = device))

	model = model.to(device)
	test_dataset = MyDataset(img_dir = arg.img_dir, anno_dir = arg.anno_dir, phase = 'test')
	n_test_img = len(test_dataset)
	test_loader = DataLoader(test_dataset, batch_size = arg.batch, 
				shuffle = True, pin_memory = True, num_workers = arg.num_workers)
	model.eval()
	with torch.no_grad():
		img, anno_path_list, p_palette = iter(test_loader).next()
		# img.size(): (batch, 3, 475, 475), len(anno_path_list) = batch 	
		print('input image')
		unnormalize_show(img[0], arg.save_dir)
		
		print('ground truth')
		anno_path = anno_path_list[0]
		PIL_show(anno_path, arg.save_dir, area = (274, 18, 749, 493))
		
		img = img.to(device)
		outputs = model(img)
		
		# the other output is ignored for inference
		# pred.size(): (batch, n_classes, 475, 475)
		pred, _ = outputs
		
		# select highest probability in n_classes axis(dim)
		pred = torch.argmax(pred[0], dim = 0)
		pred = pred.cpu().detach().numpy()
		
		# pred.shape: (475, 475)
		PIL_pred = Image.fromarray(np.uint8(pred), mode = 'P')
		
		p_palette = (p_palette).tolist()[0]
		PIL_pred.putpalette(p_palette)
		print('predicted image')
		PIL_pred.show()
		if arg.save_dir is not None:
			PIL_pred.save(f'{arg.save_dir}output.png')
			print(f'save img in {arg.save_dir}')

