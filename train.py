import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from time import time

from model import PSPNet
from loss import PSPLoss
from dataset import MyDataset
from config import Config, load_pkl, train_parser

# ~ https://github.com/fregu856/deeplabv3/blob/master/train.py

def train(cfg):
	torch.backends.cudnn.benchmark = True	
	device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
	model = PSPNet(n_classes = cfg.n_classes)
	model = model.to(device)
	
	train_dataset = MyDataset(img_dir = cfg.img_dir, anno_dir = cfg.anno_dir, phase = 'train')
	n_train_img = len(train_dataset)
	val_dataset = MyDataset(img_dir = cfg.img_dir, anno_dir = cfg.anno_dir, phase = 'val')
	n_val_img = len(val_dataset)
	train_loader = DataLoader(train_dataset, batch_size = cfg.batch, 
				shuffle = True, pin_memory = True, num_workers = cfg.num_workers)
	val_loader = DataLoader(val_dataset, batch_size = cfg.batch, 
				shuffle = False, pin_memory = True, num_workers = cfg.num_workers) 
	loss_func = PSPLoss(aux_weight = 0.4)
	# ~ optimizer = optim_func(model)
	optimizer = optim.Adam(model.parameters(), lr = cfg.lr)
	# ~ scheduler = schedule_func(optimizer)
	min_val_avg_loss, cnt = 1e4, 0
	
	for epoch in range(1, cfg.epochs+1):
		
		train_avg_loss = 0.
		model.train()
		t1 = time()
		for t, (img, anno_img) in enumerate(train_loader, 1):	
			if img.size(0) != cfg.batch:
				break# to avoid batch normalization error
			img = img.to(device)
			anno_img = anno_img.to(device)
			outputs = model(img)
			loss = loss_func(outputs, anno_img.long())
			optimizer.zero_grad()
			loss.backward()
			optimizer.step()
			# ~ scheduler.step()
			
			train_avg_loss += loss.item()
			
			if t % (cfg.batch_verbose) == 0:
				t2 = time()
				print('train Epoch: %d, batch: %d, sample: %d/%d, Avg Loss: %1.3f, %dmin%dsec'%(
					epoch, t, t*cfg.batch, n_train_img, train_avg_loss/t, (t2-t1)//60, (t2-t1)%60))
				if cfg.islogger:
					if t == cfg.batch_verbose and epoch == 1:
						train_csv_path = '%s%s_train.csv'%(cfg.log_dir, cfg.dump_date)#cfg.log_dir = ./Csv/
						print(f'generate {train_csv_path}')
						with open(train_csv_path, 'w') as f:
							f.write('time,epoch,batch,sample,train avg loss\n')
					with open(train_csv_path, 'a') as f:
						f.write('%dmin%dsec,%d,%d,%d,%1.3f\n'%(
							(t2-t1)//60, (t2-t1)%60, epoch, t, t*cfg.batch, train_avg_loss/t))
				t1 = time()	
			
		val_avg_loss = 0.
		model.eval()
		t3 = time()
		with torch.no_grad():
			for t, (img, anno_img) in enumerate(val_loader, 1):
				if img.size(0) != cfg.batch:
					break# to avoid batch normalization error
					
				img = img.to(device)
				anno_img = anno_img.to(device)
				outputs = model(img)
				loss = loss_func(outputs, anno_img.long())
				val_avg_loss += loss.item()
		
				if t % (cfg.batch_verbose) == 0:
					t4 = time()
					print('val Epoch: %d, batch: %d, sample: %d/%d, Avg Loss: %1.3f, %dmin%dsec'%(
						epoch, t, t*cfg.batch, n_val_img, val_avg_loss/t, (t4-t3)//60, (t4-t3)%60))
					"""
					if cfg.islogger:
						if t == cfg.batch_verbose and epoch == 1:
							log_path = '%s%s_val.csv'%(cfg.log_dir, cfg.dump_date)#cfg.log_dir = ./Csv/
							print(f'generate {log_path}')
							with open(log_path, 'w') as f:
								f.write('time,epoch,batch,sample,loss\n')
						with open(log_path, 'a') as f:
							f.write('%dmin%dsec,%d,%d,%d, %1.3f\n'%(
								(t4-t3)//60, (t4-t3)%60, epoch, t, t*cfg.batch, val_avg_loss/t))
					"""
					t3 = time()
			
		if epoch == 1:
			param_path = '%s%s_param.csv'%(cfg.log_dir, cfg.dump_date)# cfg.log_dir = ./Csv/
			print(f'generate {param_path}')
			with open(param_path, 'w') as f:
				f.write(''.join('%s,%s\n'%item for item in vars(cfg).items())) 
				f.write('n_train_img,%d\n'%(n_train_img))
				f.write('n_val_img,%d\n'%(n_val_img))
		
		if val_avg_loss/t < min_val_avg_loss:
			min_val_avg_loss = val_avg_loss/t
			print('update min_val_avg_loss: ', min_val_avg_loss)
			
			
			weight_path = '%s%s_epoch%s.pt'%(cfg.weight_dir, cfg.dump_date, epoch)
			torch.save(model.state_dict(), weight_path)
			print(f'generate {weight_path}')
			
			if cfg.islogger:
				if epoch == 1:
					val_csv_path = '%s%s_val.csv'%(cfg.log_dir, cfg.dump_date)#cfg.log_dir = ./Csv/
					print(f'generate {val_csv_path}')
					with open(val_csv_path, 'w') as f:
						f.write('time,epoch,val avg loss\n')
				with open(val_csv_path, 'a') as f:
					f.write('%dmin%dsec,%d,%1.3f\n'%(
						(t4-t3)//60, (t4-t3)%60, epoch, val_avg_loss/t))
			
		else:# val_avg_loss/t >= min_val_avg_loss:
			cnt += 1
			print(f'current cnt: {cnt}/7')
			if cnt >= 7:
				print('early stopping')
				break
			
if __name__ == '__main__':
	cfg = load_pkl(train_parser().path)
	train(cfg)	
