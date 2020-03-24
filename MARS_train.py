from dataset.dataset import *
from torch.utils.data import Dataset, DataLoader
import getpass
import os
import socket
import numpy as np
from dataset.preprocess_data import *
from PIL import Image, ImageFilter
import argparse
import torch
from torch import nn
from torch import optim
from torch.optim import lr_scheduler
from models.model import generate_model
from opts import parse_opts
from torch.autograd import Variable
import time
import sys
from utils import *
#from utils import AverageMeter, calculate_accuracy
import pdb
import math

def sigmoid(x):
  return 1 / (1 + math.exp(-x))

def kl_loss(pre1,pre2):
	criterion_softmax = torch.nn.Softmax(dim=1).cuda()
	pre1=criterion_softmax(pre1)
	pre2=criterion_softmax(pre2)
	loss=torch.mean(torch.sum(pre2*torch.log(1e-8+pre2/(pre1+1e-8)),1))
	return loss

if __name__=="__main__":
	opt = parse_opts()
	print(opt)

	opt.arch = '{}-{}'.format(opt.model, opt.model_depth)
	torch.manual_seed(opt.manual_seed)

	print("Preprocessing train data ...")
	train_data   = globals()['{}_test'.format(opt.dataset)](split = opt.split, train = 1, opt = opt)
	print("Length of train data = ", len(train_data))

	print("Preprocessing validation data ...")
	val_data   = globals()['{}_test'.format(opt.dataset)](split = opt.split, train = 2, opt = opt)
	print("Length of validation data = ", len(val_data))

	if opt.modality=='RGB': opt.input_channels = 3
	elif opt.modality=='Flow': opt.input_channels = 2

	print("Preparing datatloaders ...")
	train_dataloader = DataLoader(train_data, batch_size = opt.batch_size, shuffle=True, num_workers = opt.n_workers, pin_memory = True, drop_last=True)
	val_dataloader   = DataLoader(val_data, batch_size = opt.batch_size, shuffle=True, num_workers = opt.n_workers, pin_memory = True, drop_last=True)
	print("Length of train datatloader = ",len(train_dataloader))
	print("Length of validation datatloader = ",len(val_dataloader))

	log_path_MARS = os.path.join(opt.result_path_MARS, opt.dataset)
	log_path_Flow = os.path.join(opt.result_path_Flow, opt.dataset)
	if not os.path.exists(log_path_MARS):
		os.makedirs(log_path_MARS)
	if not os.path.exists(log_path_Flow):
		os.makedirs(log_path_Flow)

	if opt.log == 1:
		if opt.pretrain_path != '':
			epoch_logger = Logger_MARS(os.path.join(log_path_MARS, 'PreKin_MARS_{}_{}_train_batch{}_sample{}_clip{}_lr{}_nesterov{}_manualseed{}_model{}{}_ftbeginidx{}_layer{}_alpha{}.log'
				.format(opt.dataset, opt.split, opt.batch_size, opt.sample_size, opt.sample_duration, opt.learning_rate, opt.nesterov, opt.manual_seed, opt.model, opt.model_depth, opt.ft_begin_index,
								opt.output_layers[0], opt.MARS_alpha))
							,['epoch', 'loss', 'loss_MSE', 'loss_MARS', 'acc', 'lr'], opt.MARS_resume_path, opt.begin_epoch)
			val_logger   = Logger_MARS(os.path.join(log_path_MARS, 'PreKin_MARS_{}_{}_val_batch{}_sample{}_clip{}_lr{}_nesterov{}_manualseed{}_model{}{}_ftbeginidx{}_layer{}_alpha{}.log'
							.format(opt.dataset,opt.split,  opt.batch_size, opt.sample_size, opt.sample_duration, opt.learning_rate, opt.nesterov, opt.manual_seed, opt.model, opt.model_depth, opt.ft_begin_index,
								opt.output_layers[0], opt.MARS_alpha))
							,['epoch', 'loss', 'acc'], opt.MARS_resume_path, opt.begin_epoch)
		else:
			epoch_logger_MARS = Logger_MARS(os.path.join(log_path_MARS, 'MARS_{}_{}_train_batch{}_sample{}_clip{}_lr{}_nesterov{}_manualseed{}_model{}{}_ftbeginidx{}_layer{}_alpha{}.log'
							.format(opt.dataset, opt.split, opt.batch_size, opt.sample_size, opt.sample_duration, opt.learning_rate, opt.nesterov, opt.manual_seed, opt.model, opt.model_depth, opt.ft_begin_index,
								opt.output_layers[0], opt.MARS_alpha))
							,['epoch', 'loss_MARS', 'loss_MARS_pre', 'loss_MARS', 'acc_MARS', 'lr_MARS'], opt.MARS_resume_path, opt.begin_epoch)

			epoch_logger_Flow = Logger_MARS(os.path.join(log_path_Flow,
													'Flow_{}_{}_train_batch{}_sample{}_clip{}_lr{}_nesterov{}_manualseed{}_model{}{}_ftbeginidx{}_layer{}_alpha{}.log'
													.format(opt.dataset, opt.split, opt.batch_size, opt.sample_size,
															opt.sample_duration, opt.learning_rate, opt.nesterov,
															opt.manual_seed, opt.model, opt.model_depth,
															opt.ft_begin_index,
															opt.output_layers[0], opt.MARS_alpha))
											, ['epoch', 'loss_Flow', 'loss_Flow_pre', 'acc_Flow', 'lr_Flow'],
											opt.MARS_resume_path, opt.begin_epoch)
			val_logger_MARS   = Logger_MARS(os.path.join(log_path_MARS, 'MARS_{}_{}_val_batch{}_sample{}_clip{}_lr{}_nesterov{}_manualseed{}_model{}{}_ftbeginidx{}_layer{}_alpha{}.log'
							.format(opt.dataset, opt.split, opt.batch_size, opt.sample_size, opt.sample_duration, opt.learning_rate, opt.nesterov, opt.manual_seed, opt.model, opt.model_depth, opt.ft_begin_index,
								opt.output_layers[0], opt.MARS_alpha))
							,['epoch', 'loss', 'acc'], opt.MARS_resume_path, opt.begin_epoch)
			val_logger_Flow = Logger_MARS(os.path.join(log_path_Flow,
												  'Flow_{}_{}_val_batch{}_sample{}_clip{}_lr{}_nesterov{}_manualseed{}_model{}{}_ftbeginidx{}_layer{}_alpha{}.log'
												  .format(opt.dataset, opt.split, opt.batch_size, opt.sample_size,
														  opt.sample_duration, opt.learning_rate, opt.nesterov,
														  opt.manual_seed, opt.model, opt.model_depth,
														  opt.ft_begin_index,
														  opt.output_layers[0], opt.MARS_alpha))
									 , ['epoch', 'loss', 'acc'], opt.MARS_resume_path, opt.begin_epoch)

	if opt.pretrain_path!='' and opt.dataset!= 'Kinetics':
		opt.weight_decay = 1e-5
		opt.learning_rate = 0.001

	if opt.nesterov: dampening = 0
	else: dampening = opt.dampening


	# define the model
	print("Loading MARS model... ", opt.model, opt.model_depth)
	opt.input_channels =3
	model_MARS, parameters_MARS = generate_model(opt)
	print("Loading Flow model... ", opt.model, opt.model_depth)
	opt.input_channels = 2
	model_Flow,parameters_Flow=generate_model(opt)

	if opt.pretrain_path != '':
		opt.pretrain_path = ''
		if opt.dataset == 'HMDB51':
			opt.n_classes = 51
		elif opt.dataset == 'Kinetics':
			opt.n_classes = 400
		elif opt.dataset == 'UCF101':
			opt.n_classes = 101

	criterion_cross = nn.CrossEntropyLoss().cuda()
	criterion_softmax=torch.nn.Softmax(dim=1).cuda()

	if opt.resume_path1:
		print('loading checkpoint {}'.format(opt.resume_path1))
		checkpoint = torch.load(opt.resume_path1)
		model_Flow.load_state_dict(checkpoint['state_dict'])



	if opt.MARS_resume_path:
		print('loading MARS checkpoint {}'.format(opt.MARS_resume_path))
		checkpoint = torch.load(opt.MARS_resume_path)
		assert opt.arch == checkpoint['arch']

		opt.begin_epoch = checkpoint['epoch']
		model_MARS.load_state_dict(checkpoint['state_dict'])


	print("Initializing the optimizer ...")

	print("lr = {} \t momentum = {} \t dampening = {} \t weight_decay = {}, \t nesterov = {}"
				.format(opt.learning_rate, opt.momentum, dampening, opt. weight_decay, opt.nesterov))
	print("LR patience = ", opt.lr_patience)

	optimizer_MARS= optim.SGD(
		parameters_MARS,
		lr=opt.learning_rate,
		momentum=opt.momentum,
		dampening=dampening,
		weight_decay=opt.weight_decay,
		nesterov=opt.nesterov)
	scheduler_MARS = lr_scheduler.ReduceLROnPlateau(optimizer_MARS, 'min', patience=opt.lr_patience)

	optimizer_Flow= optim.SGD(
		parameters_Flow,
		lr=opt.learning_rate,
		momentum=opt.momentum,
		dampening=dampening,
		weight_decay=opt.weight_decay,
		nesterov=opt.nesterov)
	scheduler_Flow = lr_scheduler.ReduceLROnPlateau(optimizer_Flow, 'min', patience=opt.lr_patience)

	if opt.MARS_resume_path != '':
		print("Loading optimizer checkpoint state")
		optimizer.load_state_dict(torch.load(opt.MARS_resume_path)['optimizer'])

   

	print('run')
	for epoch in range(opt.begin_epoch, opt.n_epochs + 1):
		model_Flow.train()
		model_MARS.train()
		batch_time = AverageMeter()
		data_time = AverageMeter()
		losses_MARS_cross = AverageMeter()
		losses_Flow_cross = AverageMeter()
		losses_Flow_pre = AverageMeter()
		losses_MARS_pre= AverageMeter()
		losses_Flow = AverageMeter()
		losses_MARS = AverageMeter()
		accuracies_MARS = AverageMeter()
		accuracies_Flow = AverageMeter()

		end_time = time.time()
		for i, (inputs, targets) in enumerate(train_dataloader):
			data_time.update(time.time() - end_time)
			inputs_MARS  = inputs[:,0:3,:,:,:]
			inputs_Flow = inputs[:,3:,:,:,:]


			targets = targets.cuda(non_blocking=True)
			# pdb.set_trace()
			inputs_MARS  = Variable(inputs_MARS)
			inputs_Flow = Variable(inputs_Flow)
			targets     = Variable(targets)
			outputs_MARS  = model_MARS(inputs_MARS)
			outputs_Flow = model_Flow(inputs_Flow)

			######

			""""
			Bat_MARS = nn.BatchNorm2d(outputs_MARS[1].shape[1]).cuda()
			Bat_Flow = nn.BatchNorm2d(outputs_Flow.shape[1]).cuda()

			outputs_MARS1 = outputs_MARS[1].reshape(outputs_MARS[1].shape[0], outputs_MARS[1].shape[1], 1, 1)
			outputs_Flow1 = outputs_Flow.reshape(outputs_Flow.shape[0], outputs_Flow.shape[1], 1, 1)

			outputs_MARS1 = Bat_MARS(outputs_MARS1)
			outputs_Flow1 = Bat_Flow(outputs_Flow1)

			outputs_MARS1 = outputs_MARS1.reshape(outputs_MARS1.shape[0], outputs_MARS1.shape[1])
			outputs_Flow1 = outputs_Flow1.reshape(outputs_Flow1.shape[0], outputs_Flow1.shape[1])

			ddd = criterion_Flow(outputs_MARS1, outputs_Flow)
			"""

			######
			### Original version Loss of author
			loss_MARS_cross = criterion_cross(outputs_MARS[0], targets)
			loss_MARS_pre=kl_loss(outputs_MARS[0],outputs_Flow[0])
			loss_MARS=loss_MARS_cross+opt.MARS_alpha*loss_MARS_pre
			acc_MARS = calculate_accuracy(outputs_MARS[0], targets)
			losses_MARS.update(loss_MARS.data, inputs.size(0))
			losses_MARS_pre.update(loss_MARS_pre.data, inputs.size(0))
			accuracies_MARS.update(acc_MARS, inputs.size(0))
			optimizer_MARS.zero_grad()
			loss_MARS.backward(retain_graph=True)
			optimizer_MARS.step()

			#flow_loss
			loss_Flow_cross = criterion_cross(outputs_Flow[0], targets)
			loss_Flow_pre=kl_loss(outputs_Flow[0],outputs_MARS[0])
			loss_Flow=loss_Flow_cross+opt.MARS_alpha*loss_Flow_pre
			acc_Flow = calculate_accuracy(outputs_Flow[0], targets)
			losses_Flow.update(loss_Flow.data, inputs.size(0))
			losses_Flow_pre.update(loss_Flow_pre.data, inputs.size(0))
			accuracies_Flow.update(acc_Flow, inputs.size(0))

			optimizer_Flow.zero_grad()
			loss_Flow.backward()
			optimizer_Flow.step()
			batch_time.update(time.time() - end_time)
			end_time = time.time()

			print('Epoch: [{0}][{1}/{2}]\t'
				  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
				  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
				  'acc_MARS {acc_MARS.val:.4f} ({acc_MARS.avg:.4f})\t'
				  'Loss_MARS {loss_MARS.val:.4f} ({loss_MARS.avg:.4f})\t'
				  'acc_Flow {acc_Flow.val:.3f} ({acc_Flow.avg:.3f})\t'
				  'Loss_Flow {loss_Flow.val:.4f} ({loss_Flow.avg:.4f})'.format(
					  epoch,
					  i + 1,
					  len(train_dataloader),
					  batch_time=batch_time,
					  data_time=data_time,
					  acc_MARS=accuracies_MARS,
					  loss_MARS=losses_MARS,
					  acc_Flow=accuracies_Flow,
				      loss_Flow=losses_Flow))

		if opt.log == 1:
			epoch_logger_MARS.log({
				'epoch': epoch,
				'loss_MARS': losses_MARS.avg,
				'losses_MARS_cross':losses_MARS_cross.avg,
				'loss_MARS_pre':losses_MARS_pre.avg,
				'acc_MARS': accuracies_MARS.avg,
				'lr_MARS': optimizer_MARS.param_groups[0]['lr']

			})
		if opt.log == 1:
			epoch_logger_Flow.log({
				'epoch': epoch,
				'loss_Flow': losses_Flow.avg,
				'loss_Flow_cross':losses_Flow_cross.avg,
				'loss_Flow_pre': losses_Flow_pre.avg,
				'acc_Flow': accuracies_Flow.avg,
				'lr_Flow': optimizer_Flow.param_groups[0]['lr']

			})
		if epoch % opt.checkpoint == 0:
			if opt.pretrain_path != '':
				save_file_path_MARS = os.path.join(log_path, 'MARS_preKin_{}_{}_train_batch{}_sample{}_clip{}_lr{}_nesterov{}_manualseed{}_model{}{}_ftbeginidx{}_layer{}_alpha{}_{}.pth'
							.format(opt.dataset, opt.split, opt.batch_size, opt.sample_size, opt.sample_duration, opt.learning_rate, opt.nesterov, opt.manual_seed, opt.model, opt.model_depth, opt.ft_begin_index,
								opt.output_layers[0], opt.MARS_alpha, epoch))
				save_file_path_Flow = os.path.join(log_path,
												   'Flow_preKin_{}_{}_train_batch{}_sample{}_clip{}_lr{}_nesterov{}_manualseed{}_model{}{}_ftbeginidx{}_layer{}_alpha{}_{}.pth'
												   .format(opt.dataset, opt.split, opt.batch_size, opt.sample_size,
														   opt.sample_duration, opt.learning_rate, opt.nesterov,
														   opt.manual_seed, opt.model, opt.model_depth,
														   opt.ft_begin_index,
														   opt.output_layers[0], opt.MARS_alpha, epoch))
			else:
				save_file_path_MARS = os.path.join(log_path_MARS, 'MARS_{}_{}_train_batch{}_sample{}_clip{}_lr{}_nesterov{}_manualseed{}_model{}{}_ftbeginidx{}_layer{}_alpha{}_{}.pth'
							.format(opt.dataset, opt.split, opt.batch_size, opt.sample_size, opt.sample_duration, opt.learning_rate, opt.nesterov, opt.manual_seed, opt.model, opt.model_depth, opt.ft_begin_index,
								opt.output_layers[0], opt.MARS_alpha, epoch))
				save_file_path_Flow= os.path.join(log_path_Flow,
												   'Flow_{}_{}_train_batch{}_sample{}_clip{}_lr{}_nesterov{}_manualseed{}_model{}{}_ftbeginidx{}_layer{}_alpha{}_{}.pth'
												   .format(opt.dataset, opt.split, opt.batch_size, opt.sample_size,
														   opt.sample_duration, opt.learning_rate, opt.nesterov,
														   opt.manual_seed, opt.model, opt.model_depth,
														   opt.ft_begin_index,
														   opt.output_layers[0], opt.MARS_alpha, epoch))
			states_MARS = {
				'epoch': epoch + 1,
				'arch': opt.arch,
				'state_dict': model_MARS.state_dict(),
				'optimizer_MARS': optimizer_MARS.state_dict(),
			}
			torch.save(states_MARS, save_file_path_MARS)
			states_Flow = {
				'epoch': epoch + 1,
				'arch': opt.arch,
				'state_dict': model_Flow.state_dict(),
				'optimizer_Flow': optimizer_Flow.state_dict(),
			}
			torch.save(states_Flow, save_file_path_Flow)

		model_MARS.eval()
		model_Flow.eval()

		batch_time = AverageMeter()
		data_time = AverageMeter()
		losses_MARS = AverageMeter()
		losses_Flow=AverageMeter()
		accuracies_Flow=AverageMeter()
		accuracies_MARS = AverageMeter()

		end_time = time.time()
		with torch.no_grad():
			for i, (inputs, targets) in enumerate(val_dataloader):

				data_time.update(time.time() - end_time)
				inputs_MARS  = inputs[:,0:3,:,:,:]
				inputs_Flow = inputs[:, 3:, :, :, :]

				targets = targets.cuda(non_blocking=True)
				inputs_MARS  = Variable(inputs_MARS)
				inputs_Flow = Variable(inputs_Flow)
				targets     = Variable(targets)

				outputs_MARS  = model_MARS(inputs_MARS)
				outputs_Flow = model_Flow(inputs_Flow)

				loss_MARS_cross = criterion_cross(outputs_MARS[0], targets)
				acc_MARS  = calculate_accuracy(outputs_MARS[0], targets)

				losses_MARS.update(loss_MARS_cross.data, inputs.size(0))
				accuracies_MARS.update(acc_MARS, inputs.size(0))
				#######flow
				loss_Flow_cross=criterion_cross(outputs_Flow[0],targets)
				acc_Flow=calculate_accuracy(outputs_Flow[0], targets)
				losses_Flow.update(loss_Flow_cross.data,inputs.size(0))
				accuracies_Flow.update(acc_Flow,inputs.size(0))

				batch_time.update(time.time() - end_time)
				end_time = time.time()

				print('Val_Epoch: [{0}][{1}/{2}]\t'
					  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
					  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
					  'acc_MARS {acc_MARS.val:.4f} ({acc_MARS.avg:.4f})\t'
					  'Loss_MARS {loss_MARS.val:.4f} ({loss_MARS.avg:.4f})\t'
					  'acc_Flow {acc_Flow.val:.3f} ({acc_Flow.avg:.3f})\t'
					  'Loss_Flow {loss_Flow.val:.4f} ({loss_Flow.avg:.4f})'.format(
						epoch,
						i + 1,
						len(val_dataloader),
						batch_time=batch_time,
					    data_time=data_time,
					    acc_MARS=accuracies_MARS,
					    loss_MARS=losses_MARS,
					    acc_Flow=accuracies_Flow,
					    loss_Flow=losses_Flow))

		if opt.log == 1:
			val_logger_MARS.log({'epoch': epoch, 'loss': losses_MARS.avg, 'acc': accuracies_MARS.avg})
			val_logger_Flow.log({'epoch': epoch, 'loss': losses_Flow.avg, 'acc': accuracies_Flow.avg})

		scheduler_MARS.step(losses_MARS.avg)
		scheduler_Flow.step(losses_Flow.avg)



