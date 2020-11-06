import pickle
import os
import argparse
from datetime import datetime

def arg_parser():
	parser = argparse.ArgumentParser()
	parser.add_argument('--seed', metavar = 'SE', type = int, help = 'seed random number for inference, reproducibility')
	# main config
	parser.add_argument('-c', '--n_classes', metavar = 'C', type = int, default = 35, help = 'number of classes for output')
	parser.add_argument('-b', '--batch', metavar = 'B', type = int, default = 4, help = 'batch size')
	parser.add_argument('-v', '--batch_verbose', metavar = 'V', type = int, default = 50, help = 'print and log during training')
	parser.add_argument('-n', '--num_workers', metavar = 'N', type = int, default = 2, help = 'args num_workers in Dataloader, pytorch')
	parser.add_argument('-e', '--epochs', metavar = 'E', type = int, default = 200, help = 'total number of samples = epochs * available imgs')
	parser.add_argument('--lr', metavar = 'LR', type = float, default = 1e-4, help = 'initial learning rate')
	
	parser.add_argument('--islogger', action = 'store_false', help = 'flag for csv logger, default true')
	parser.add_argument('-ld', '--log_dir', metavar = 'LD', type = str, default = './Csv/', help = 'csv logger dir')
	parser.add_argument('-id', '--img_dir', metavar = 'ID', type = str, default = './data/leftImg8bit/', help = 'images dir')
	parser.add_argument('-ad', '--anno_dir', metavar = 'AD', type = str, default = './data/gtFine/', help = 'annotation images (ground truth) dir')
	parser.add_argument('-wd', '--weight_dir', metavar = 'MD', type = str, default = './Weights/', help = 'model weight save dir')
	parser.add_argument('-pd', '--pkl_dir', metavar = 'PD', type = str, default = './Pkl/', help = 'pkl save dir')
	parser.add_argument('-cd', '--cuda_dv', metavar = 'DV', type = str, default = '0', help = 'os CUDA_VISIBLE_DEVICE')
	args = parser.parse_args()
	return args

class Config():
	def __init__(self, **kwargs):	
		for k, v in kwargs.items():
			self.__dict__[k] = v
		self.mode = 'train'
		self.optimizer = 'Adam'
		self.dump_date = datetime.now().strftime('%m%d_%H_%M')
		for x in [self.log_dir, self.weight_dir, self.pkl_dir]:
			os.makedirs(x, exist_ok = True)
		self.pkl_path = self.pkl_dir + self.dump_date + '.pkl'
		
def print_cfg(cfg):
	print(''.join('%s: %s\n'%item for item in vars(cfg).items()))
		
def dump_pkl(args, verbose = True):
	cfg = Config(**vars(args))
	with open(cfg.pkl_path, 'wb') as f:
		pickle.dump(cfg, f)
		print('--- save pickle file at: %s ---\n'%cfg.pkl_path)
		if verbose:
			print_cfg(cfg)
			
def load_pkl(pkl_path, verbose = True):
	if not os.path.isfile(pkl_path):
		raise FileNotFoundError('pkl_path')
	with open(pkl_path, 'rb') as f:
		cfg = pickle.load(f)
		os.environ['CUDA_VISIBLE_DEVICE'] = cfg.cuda_dv
		if verbose:
			print_cfg(cfg)
	return cfg

def train_parser():
	parser = argparse.ArgumentParser()
	parser.add_argument('-p', '--path', metavar = 'P', type = str, 
						required = True, help = 'Pkl/***.pkl, pkl file only')
	args = parser.parse_args()
	return args

def test_parser():
	parser = argparse.ArgumentParser()
	parser.add_argument('-p', '--path', metavar = 'P', type = str, required = True,  
						help = 'Weights/***.pt, pt file required')
	parser.add_argument('-c', '--n_classes', metavar = 'C', type = int, default = 35, help = 'number of classes for output')
	parser.add_argument('-n', '--num_workers', metavar = 'N', type = int, default = 2, help = 'args num_workers in Dataloader, pytorch')
	parser.add_argument('-b', '--batch', metavar = 'B', type = int, default = 2, help = 'batch size')
	parser.add_argument('-id', '--img_dir', metavar = 'ID', type = str, default = './data/leftImg8bit/', help = 'images dir')
	parser.add_argument('-ad', '--anno_dir', metavar = 'AD', type = str, default = './data/gtFine/', help = 'annotation images (ground truth) dir')
	parser.add_argument('-sd', '--save_dir', metavar = 'SD', type = str, default = './SaveImg/', help = 'save output and input image dir')
	parser.add_argument('-s', '--seed', metavar = 'S', type = int, default = 123, help = 'random seed number for inference, reproducibility')
	
	args = parser.parse_args()
	return args

if __name__ == '__main__':
	args = arg_parser()
	dump_pkl(args)
	# cfg = load_pkl(file_parser().path)
	# for k, v in vars(cfg).items():
	# 	print(k, v)
	# 	print(vars(cfg)[k])#==v
