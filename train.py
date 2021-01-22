import torch.distributed as dist

import time
import datetime

import os
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
from torch.autograd import Variable, grad
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from model import *
from dataset import *
from util import *

import itertools
import matplotlib.pyplot as plt

from torchvision import datasets, transforms, utils

MEAN = 0.5
STD = 0.5

num_workers = 0


def train(gpu, ngpus_per_node, args):
    ## PGGAN
    n_label = 1
    code_size = 512 - n_label
    n_critic = 1 # 얘 지워도 되지 않나? ㅇㅇ

    args.gpu = gpu
    ngpus_per_node = torch.cuda.device_count()
    print("Use GPU: {} for training".format(args.gpu))

    args.rank = args.rank * ngpus_per_node + gpu
    dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                            world_size=args.world_size, rank=args.rank)

    ## 트레이닝 파라메터 설정하기
    mode = args.mode
    train_continue = args.train_continue

    lr = args.lr
    batch_size = int(args.batch_size / ngpus_per_node)
    num_workers = int(args.num_workers / ngpus_per_node)
    num_epoch = args.num_epoch

    data_dir = args.data_dir
    ckpt_dir = args.ckpt_dir
    log_dir = args.log_dir
    result_dir = args.result_dir

    task = args.task
    opts = [args.opts[0], np.asarray(args.opts[1:]).astype(np.float)]

    ny = args.ny
    nx = args.nx
    nch = args.nch
    nker = args.nker

    wgt_cycle = args.wgt_cycle
    wgt_ident = args.wgt_ident
    norm = args.norm

    network = args.network
    learning_type = args.learning_type

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print("mode: %s" % mode)
    print("norm: %s" % norm)

    print("learning rate: %.4e" % lr)
    print("batch size: %d" % batch_size)
    print("number of epoch: %d" % num_epoch)

    print("task: %s" % task)
    print("opts: %s" % opts)

    print("network: %s" % network)
    print("learning type: %s" % learning_type)

    print("data dir: %s" % data_dir)
    print("ckpt dir: %s" % ckpt_dir)
    print("log dir: %s" % log_dir)
    print("result dir: %s" % result_dir)

    print("device: %s" % device)

    ## 디렉토리 생성하기
    result_dir_train = os.path.join(result_dir, 'train')

    ## 네트워크 학습하기
    if mode == 'train':

        def loader_train(transform):
            dataset_train = datasets.ImageFolder(data_dir,transform=transform)
            train_sampler = torch.utils.data.distributed.DistributedSampler(dataset_train)

            loader_train = DataLoader(dataset_train, batch_size=batch_size,
                                      shuffle=(train_sampler is None), num_workers=num_workers,
                                      sampler=train_sampler)

            # loader_train = DataLoader(dataset_train, shuffle=True, batch_size=batch_size, num_workers=num_workers)
            return loader_train

        def sample_data(loader_train, image_size=4):
            transform = transforms.Compose([
                transforms.Resize(image_size),
                transforms.CenterCrop(image_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=MEAN, std=STD)
            ])

            loader_train = loader_train(transform)

            for img, label in loader_train:
                yield img, label

        def requires_grad(model, flag=True):
            for p in model.parameters():
                p.requires_grad = flag

        def accumulate(model1, model2, decay=0.999):
            par1 = dict(model1.named_parameters())
            par2 = dict(model2.named_parameters())

            for k in par1.keys():
                par1[k].data.mul_(decay).add_(1-decay, par2[k].data)

        # 그밖에 부수적인 variables 설정하기
        # num_data_train = len(datasㅅet_train)    왜 얘안되냐
        # num_batch_train = np.ceil(num_data_train / batch_size)

    ## 네트워크 생성하기
    if network == "PGGAN":
        print('==> Making model..')
        netG = PGGAN(code_size,n_label)
        netG_running = PGGAN(code_size,n_label)
        netG_running.train(False)
        netD = Discriminator(n_label)

        ## 이 초기화 위치 여기 맞나
        accumulate(netG_running, netG, 0)

        torch.cuda.set_device(args.gpu)
        netG.cuda(args.gpu)
        netG_running.cuda(args.gpu)
        netD.cuda(args.gpu)

        netG = torch.nn.parallel.DistributedDataParallel(netG, device_ids=[args.gpu],find_unused_parameters=True)
        netG_running = torch.nn.parallel.DistributedDataParallel(netG_running, device_ids=[args.gpu],find_unused_parameters=True)
        netD = torch.nn.parallel.DistributedDataParallel(netD, device_ids=[args.gpu],find_unused_parameters=True)

        netG_num_params = sum(p.numel() for p in netG.parameters() if p.requires_grad)
        netG_running_num_params = sum(p.numel() for p in netG_running.parameters() if p.requires_grad)
        netD_num_params = sum(p.numel() for p in netD.parameters() if p.requires_grad)

        print('The number of netG parameters of model is', netG_num_params)
        print('The number of netG_running parameters of model is', netG_running_num_params)
        print('The number of netD parameters of model is', netD_num_params)


    ## 손실함수 정의하기
    # class_loss = nn.CrossEntropyLoss()

    ## Optimizer 설정하기
    optimG = torch.optim.SGD(netG.parameters(), lr=lr, momentum=0.9, weight_decay=1e-4)
    optimD = torch.optim.SGD(netD.parameters(), lr=lr, momentum=0.9, weight_decay=1e-4)
    # optimG = torch.optim.Adam(netG.parameters(), lr=lr, betas=(0.5, 0.999))
    # optimD = torch.optim.Adam(netD.parameters(), lr=lr, betas=(0.5, 0.999))


    ## Tensorboard 를 사용하기 위한 SummaryWriter 설정
    writer_train = SummaryWriter(log_dir=os.path.join(log_dir, 'train'))

    ## 네트워크 학습시키기
    st_epoch = 0
    step = 0
    iteration = 0

    dataset_train = sample_data(loader_train, 4 * 2 ** (step))

    requires_grad(netG, False)
    requires_grad(netD, True)

    loss_G_train = []
    loss_D_train = []
    loss_grad_train = []

    alpha = 0
    one = torch.tensor(1,dtype=torch.float).to(device)
    mone = one * -1
    stabilize = False

    # TRAIN MODE
    if mode == 'train':
        if train_continue == "on":
            netG, netD, \
            optimG, optimD, st_epoch = load(ckpt_dir=ckpt_dir,
                                            netG=netG, netD=netD,
                                            optimG=optimG, optimD=optimD)

        epoch_start = time.time()
        for epoch in range(st_epoch + 1, num_epoch + 1):
            # start = time.time()

            netG.train()
            netD.train()

            netD.zero_grad()
            alpha = min(1, 2e-5 * epoch)

            if stabilize is False and iteration > 50000:
                dataset_train = sample_data(loader_train,4 * 2 ** step)
                stabilize = True
            if iteration > 100000:
                alpha = 0
                iteration = 0
                step += 1
                stabilize = False
                if step > 5:
                    alpha = 1
                    step = 5
                dataset_train = sample_data(loader_train, 4 * 2 ** step)
            try:
                real_image, label = next(dataset_train)
            except:
                dataset_train = sample_data(loader_train, 4 * 2 ** step)
                real_image, label = next(dataset_train)
            iteration += 1


            b_size = real_image.size(0)
            real_image = Variable(real_image).to(device)
            label = Variable(label).to(device)
            real_predict, real_class_predict = netD(real_image, step, alpha)
            real_predict = real_predict.mean() - 0.001 * (real_predict ** 2).mean()
            # backward안에 값을 넣으면 어떻게 되지?
            real_predict.backward(mone)

            fake_imgae = netG(Variable(torch.randn(b_size,code_size)).to(device),
                              label, step, alpha)
            fake_predict, fake_class_predict = netD(fake_imgae, step, alpha)
            fake_predict = fake_predict.mean()
            fake_predict.backward(one)

            # 해당 파트가 backward()가 있는데 forward파트가 쓰이지 않아서 계속 에러 떴었음
            # 주석처리하니까 성공
            # eps = torch.rand(b_size, 1, 1, 1).to(device)
            # x_hat = eps * real_image.data + (1 - eps) * fake_imgae.data
            # x_hat = Variable(x_hat, requires_grad=True)
            # hat_predict, _ = netD(x_hat, step, alpha)
            # grad_x_hat = grad(outputs=hat_predict.sum(), inputs=x_hat,create_graph=True)[0]
            # grad_penalty = ((grad_x_hat.view(grad_x_hat.size(0), -1)
            #                  .norm(2, dim=1) -1)**2).mean()
            # grad_penalty = 10 * grad_penalty
            # grad_penalty.backward()
            # grad_loss_val = grad_penalty.data
            # disc_loss_val = (real_predict - fake_predict).data

            optimD.step()

            ###

            ## train netG
            netG.zero_grad()
            requires_grad(netG, True)
            requires_grad(netD, False)
            # n_label = 1, code_size = 511
            input_class = Variable(torch.multinomial(torch.ones(n_label), batch_size, replacement=True)).to(device)
            fake_image = netG(Variable(torch.randn(batch_size,code_size)).to(device),
                              input_class, step, alpha)
            predict, class_predict = netD(fake_image, step, alpha)
            loss = -predict.mean()
            gen_loss_val = loss.data

            loss.backward()
            optimG.step()


            ###

            accumulate(netG_running,netG)
            requires_grad(netG, False)
            requires_grad(netD, True)

            ## save image
            if (epoch + 1) % 100 == 0:
                images = []
                for _ in range(5):
                    input_class = Variable(torch.zeros(10).long()).to(device)
                    images.append(netG_running(Variable(torch.randn(n_label*10, code_size)).to(device),
                                               input_class, step, alpha).data.cpu())
                    utils.save_image(torch.cat(images,0),f'result/{str(epoch + 1).zfill(6)}.png',
                                     nrow=n_label*10, normalize=True, range=(-1,1))
            # ## model save
            # if (epoch + 1) % 10000 == 0:
            #     torch.save(netG_running,f'checkpoint/{str(epoch+1).zfill(6)}.model')
            # print(f'{epoch + 1}; G: {gen_loss_val:.5f}; D: {disc_loss_val:.5f};'
            #      f' Grad: {grad_loss_val:.5f}; Alpha: {alpha:.3f}')



        elapse_time = time.time() - epoch_start
        elapse_time = datetime.timedelta(seconds=elapse_time)
        print("Training time {}".format(elapse_time))