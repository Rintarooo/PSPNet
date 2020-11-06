import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torchvision
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import cv2
import os

from visualize import unnormalize_show

# https://github.com/fregu856/deeplabv3/blob/master/datasets.py
# https://github.com/fregu856/segmentation/blob/master/preprocess_data.py

dirs_dict = {'train': ["jena/", "zurich/", "weimar/", "ulm/", "tubingen/", "stuttgart/",
			"strasbourg/", "monchengladbach/", "krefeld/", "hanover/",
			"hamburg/", "erfurt/", "dusseldorf/", "darmstadt/", "cologne/",
			"bremen/", "bochum/", "aachen/"],
			'val': ["frankfurt/", "munster/", "lindau/"],
			'test': ["frankfurt/", "munster/", "lindau/"],
			#'test': ["berlin/", "bielefeld/", "bonn/", "leverkusen/", "mainz/", "munich/"]
			}

# ~ ["augsburg/", "dortmund/", "freiburg/", "konigswinter/", "nuremberg/", "troisdorf/",
# ~ "bad-honnef/", "dresden/", "heidelberg/", "konstanz/", "oberhausen/", "wuppertal/",
# ~ bamberg/     duisburg/  heilbronn/   mannheim/       saarbrucken/  wurzburg/
# ~ bayreuth/    erlangen/  karlsruhe/   muhlheim-ruhr/  schweinfurt/]
# ~ dirs_dict2 = {'train': ["jena/", "zurich/", "weimar/", "ulm/", "tubingen/", "stuttgart/",
			# ~ "strasbourg/", "monchengladbach/", "krefeld/", "hanover/",
			# ~ "hamburg/", "erfurt/", "dusseldorf/", "darmstadt/", "cologne/",
			# ~ "bremen/", "bochum/", "aachen/"],
			# ~ 'val': ["frankfurt/", "munster/", "lindau/"],
			# ~ 'test': ["berlin/", "bielefeld/", "bonn/", "leverkusen/", "mainz/", "munich/"]
			# ~ }

class MyDataset(Dataset):
	def __init__(self, img_dir = 'data/leftImg8bit/', anno_dir = 'data/gtFine/', phase = 'train', h = 512, w = 1024):
		
		if phase not in dirs_dict.keys():
			raise KeyError(f'phase should be one of {dirs_dict.keys()}')
		
		if not img_dir.endswith('/'):
			img_dir = img_dir + '/'
		self.img_dir = img_dir + phase + '/'
		
		if not anno_dir.endswith('/'):
			anno_dir = anno_dir + '/'
		self.anno_dir = anno_dir + phase + '/'
		if phase == 'test':
			self.img_dir = img_dir + 'val/'
			self.anno_dir = anno_dir + 'val/'
		
		self.new_img_h = h
		self.new_img_w = w

		self.examples = []
		for city_name in dirs_dict[phase]:
			img_dir_path = self.img_dir + city_name
			file_names = os.listdir(img_dir_path)
			for file_name in file_names:
				img_id = file_name.split("_leftImg8bit.png")[0]

				img_path = self.img_dir + city_name + img_id + "_leftImg8bit.png"
				# phase == 'train' or phase == 'val'
				anno_img_path = self.anno_dir + city_name + img_id + "_gtFine_labelIds.png"
				if phase == 'test':
					anno_img_path = self.anno_dir + city_name + img_id + "_gtFine_color.png"

				example = {}
				example["img_path"] = img_path
				example["anno_img_path"] = anno_img_path
				example["img_id"] = img_id
				self.examples.append(example)

		self.num_examples = len(self.examples)
		self.phase = phase

	def __getitem__(self, idx):
		example = self.examples[idx]
		
		img = cv2.imread(example["img_path"], -1) # (shape: (1024, 2048, 3))
		assert img is not None, "cannot read img, check img_path"
		
		# resize img without interpolation (want the image to still match
		# anno_img, which we resize below):
		img = cv2.resize(img, (self.new_img_w, self.new_img_h),
				interpolation=cv2.INTER_NEAREST) # (shape: (512, 1024, 3))

		anno_img = cv2.imread(example["anno_img_path"], -1) # (shape: (1024, 2048))
		assert anno_img is not None, "cannot read anno_img, check anno_img_path"
		# resize anno_img without interpolation (want the resulting image to
		# still only contain pixel values corresponding to an object class):
		anno_img = cv2.resize(anno_img, (self.new_img_w, self.new_img_h),
						interpolation=cv2.INTER_NEAREST) # (shape: (512, 1024))

		if self.phase == 'train':
			# flip the img and the anno with 0.5 probability:
			flip = np.random.randint(low=0, high=2)
			if flip == 1:
				img = cv2.flip(img, 1)
				anno_img = cv2.flip(anno_img, 1)

			########################################################################
			# randomly scale the img and the anno:
			########################################################################
			scale = np.random.uniform(low=0.95, high=2.0)# low = 0.7
			new_img_h = int(scale*self.new_img_h)
			new_img_w = int(scale*self.new_img_w)

			# resize img without interpolation (want the image to still match
			# anno_img, which we resize below):
			img = cv2.resize(img, (new_img_w, new_img_h),
					interpolation=cv2.INTER_NEAREST) # (shape: (new_img_h, new_img_w, 3))

			# resize anno_img without interpolation (want the resulting image to
			# still only contain pixel values corresponding to an object class):
			anno_img = cv2.resize(anno_img, (new_img_w, new_img_h),
																										interpolation=cv2.INTER_NEAREST) # (shape: (new_img_h, new_img_w))
			########################################################################

			# # # # # # # # debug visualization START
			# print (scale)
			# print (new_img_h)
			# print (new_img_w)
			#
			# cv2.imshow("test", img)
			# cv2.waitKey(0)
			#
			# cv2.imshow("test", anno_img)
			# cv2.waitKey(0)
			# # # # # # # # debug visualization END

			########################################################################
			# select a 475x475 random crop from the img and anno:
			########################################################################
			start_x = np.random.randint(low=0, high=(new_img_w - 475))
			end_x = start_x + 475
			# ~ print('new_img_h', new_img_h, example["img_path"])
			start_y = np.random.randint(low=0, high=(new_img_h - 475))
			end_y = start_y + 475

			img = img[start_y:end_y, start_x:end_x] # (shape: (475, 475, 3))
			anno_img = anno_img[start_y:end_y, start_x:end_x] # (shape: (475, 475))
			########################################################################

			# # # # # # # # debug visualization START
			# print (img.shape)
			# print (anno_img.shape)
			#
			# cv2.imshow("test", img)
			# cv2.waitKey(0)
			#
			# cv2.imshow("test", anno_img)
			# cv2.waitKey(0)
			# # # # # # # # debug visualization END
		else:# phase == 'val' or phase == 'test'
			
			########################################################################
			# select a 475x475 center crop from the img and anno:
			########################################################################

			# img.shape: (512, 1024, 3)
			# ~ center_x = img.shape[1] // 2
			start_x = 274#center_x - 238
			end_x = 749#start_x + 475
			
			# ~ center_y = img.shape[0] // 2
			start_y = 18#center_y - 238
			end_y = 493#start_y + 475
			
			img = img[start_y:end_y, start_x:end_x] # (shape: (475, 475, 3))
			anno_img = anno_img[start_y:end_y, start_x:end_x] # (shape: (475, 475))
		
		# normalize the img (with the mean and std for the pretrained ResNet):
		img = img/255.0
		img = img - np.array([0.485, 0.456, 0.406])
		img = img/np.array([0.229, 0.224, 0.225]) # (shape: (475, 475, 3))
		img = np.transpose(img, (2, 0, 1)) # (shape: (3, 475, 475))
		img = img.astype(np.float32)
		
		# convert numpy -> torch:
		img = torch.from_numpy(img) # (shape: (3, 475, 475))
		anno_img = torch.from_numpy(anno_img) # (shape: (475, 475))

		if self.phase == 'test':
			left, top, right, bottom = start_x, start_y, end_x, end_y
			area = (left, top, right, bottom)
			PIL_anno_img = Image.open(example["anno_img_path"]).convert('P')
			PIL_anno_img = PIL_anno_img.crop(area)
			p_palette = np.array(PIL_anno_img.getpalette(), dtype = np.uint8)
			return img, example["anno_img_path"], p_palette
		return (img, anno_img)

	def __len__(self):
		return self.num_examples

def mono_imshow(img):
	# ~ img = img*255 #/ 2 + 0.5# unnormalize
	npimg = img.numpy()
	plt.imshow(np.transpose(npimg, (1, 2, 0)))
	plt.show()

if __name__ == '__main__':
	img_dir = 'data/leftImg8bit/'
	anno_dir = 'data/gtFine/'
	phase ='val'
	batch = 12
	num_workers = 4
	dataset = MyDataset(img_dir = img_dir, anno_dir = anno_dir, phase = phase, h = 512, w = 1024)
	print(len(dataset))
	loader = DataLoader(dataset, batch_size = batch, 
				shuffle = True, pin_memory = True, num_workers = num_workers)
	for t, (img, anno_img) in enumerate(loader, 1):
		print(img.size())
		print(anno_img.size())
		# ~ print(anno_img.long()[0])
		# ~ mono_imshow(torchvision.utils.make_grid(anno_img.unsqueeze(1)))
		unnormalize_show(torchvision.utils.make_grid(img))
		unnormalize_show(img[0])
		if t == 1:
			break
	
