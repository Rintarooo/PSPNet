import cv2
import numpy as np
import torch
import os
import matplotlib.pyplot as plt
from PIL import Image

def unnormalize_show(img, save_dir = None):
	if isinstance(img, torch.Tensor):
		img = img.numpy()
	# unnormalize in the same way you normalized in dataset Class in dataset.py(this time, inversely manipulate img)
	img = np.transpose(img, (1, 2, 0))
	img = img/np.array([1/0.229, 1/0.224, 1/0.225])
	img = img + np.array([0.485, 0.456, 0.406])
	img = img*255.0
	# convert img from BGR to RGB# https://note.nkmk.me/python-opencv-bgr-rgb-cvtcolor/ 
	img = img[:, :, ::-1]
	img = img.astype(np.uint8)
	plt.imshow(img)
	plt.show()
	if save_dir is not None:
		plt.imsave(f'{save_dir}input.png', img)

def cv_show(img_path, scale = 0.5):
	assert img_path.endswith('.png'), 'specify ***.png'
	img = cv2.imread(img_path)
	assert img is not None, 'cannot cv2 read img'
	# img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
	h, w, _ = img.shape# (1024, 2048, 3)
	img = cv2.resize(img, (int(w*scale), int(h*scale)))
	cv2.imshow(img_path, img)
	cv2.waitKey(0)
	cv2.destroyAllWindows()

def PIL_show(img_path, save_dir = None, area = None, scale = 0.5):
	assert img_path.endswith('.png'), 'specify ***.png'
	img = Image.open(img_path)
	assert img is not None, 'cannot PIL read img'
	h, w, _ = np.array(img).shape
	img = img.resize((int(w*scale), int(h*scale)))
	if area is not None:
		img = img.crop(area)
	img.show(title = img_path)
	if save_dir is not None:
		img.save(f'{save_dir}gt.png')
	

def PIL_palette(img_path, scale = 0.5):
	assert img_path.endswith('.png'), 'specify ***.png'
	img = Image.open(img_path).convert('P')#, colors = 256)
	assert img is not None, 'cannot PIL read img'
	h, w = np.array(img).shape
	img = img.resize((int(w*scale), int(h*scale)), Image.NEAREST)
	p_palette = np.array(img.getpalette(), dtype = np.uint8)#.reshape(-1, 3)
	print(p_palette)
	print(p_palette.shape)
	
if __name__ == '__main__':
	img_path = 'data/gtFine/val/lindau/lindau_000007_000019_gtFine_color.png'
	# ~ img_path = 'data/leftImg8bit/train/zurich/zurich_000002_000019_leftImg8bit.png'
	# ~ cv_show(img_path, scale = 0.5)
	PIL_show(img_path, scale = 0.5)
	# ~ PIL_palette(img_path, scale = 0.5)
