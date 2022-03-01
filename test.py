import torch.nn as nn
import torch.nn.functional as F
import torchvision
from auto_LiRPA import BoundedModule, BoundedTensor
from auto_LiRPA.perturbations import *
# import models

import torchvision.transforms as transforms
import torchvision.datasets as dsets
import sys
import time
import argparse
import pickle
import cv2


device = torch.device("cuda" if "use_cuda" else "cpu")


class FC(nn.Module):
    def __init__(self):
        super(FC, self).__init__()

        self.layer_1 = nn.Sequential(
            nn.Linear(784, 1024),
            nn.ReLU()
        )

        self.layer_2 = nn.Sequential(
            nn.Linear(1024, 1024),
            nn.ReLU()
        )

        self.layer_3 = nn.Sequential(
            nn.Linear(1024, 10),
            nn.ReLU()
        )

        # self.layer_4 = nn.Sequential(
        #     nn.Linear(1024, 10)
        # )

    def forward(self, x):
        x = x.view(-1, 28*28)

        out_1 = self.layer_1(x)
        out_2 = self.layer_2(out_1)
        out_3 = self.layer_3(out_2)
        # out_4 = self.layer_4(out_3)
        # return out_3, out_2, out_1
        return out_3

def cal_r(inputs,eps):
    grad_0 = -(inputs.grad)
    lamda = eps / torch.norm(grad_0)
    one_tensor = torch.ones(inputs.shape).to(device)
    lamda_tensor = torch.mul(lamda, one_tensor)
    r_tensor = torch.mul(lamda_tensor, grad_0)
    if torch.norm(r_tensor).data == eps:
        return r_tensor
    else:
        # print("r=",torch.norm(r_tensor).data,'eps=',eps)
        return r_tensor

def cal_r_bound(model_type,m):
    # pkl_path = f'./result/{args.model_name}/{model_type}_t{args.tao}_{args.imagestart}-{args.imagestart+args.imagetotal}_in{args.interval}.pkl'
    if model_type == 'mnist':
        test_data = torchvision.datasets.MNIST("./data", train=False, download=True,
                                               transform=torchvision.transforms.ToTensor())

        # load model
        model = FC().to(device)
        model.load_state_dict(torch.load('pretrain/FC_1024_3.pt'))
        model.eval()

    count = 0
    for ord in range(args.imagestart, args.imagestart+args.imagetotal):
        total = args.imagetotal
        is_adv = False
        if model_type == 'mnist':
            inputs = test_data[ord][0].to(device)

        inputs.requires_grad_()
        pred = model(inputs)
        orig_label = torch.argmax(pred, dim=1).cpu().numpy()
        if orig_label != test_data[ord][1]:
            print('the {} sample is not right'.format(ord))
            total -= 1
            continue
        # ord_list.append(ord)
        print('CURREET INPUT IS {} and label is {}'.format(ord,orig_label))
        sys.stdout.flush()

        tao = args.tao

        eps = 0.0000
        ptb = PerturbationLpNorm(norm=2, eps=eps)
        image = BoundedTensor(inputs, ptb)

        model = BoundedModule(model, torch.empty_like(image))
        pred = model(image)
        label = torch.argmax(pred, dim=1).cpu().numpy()
        lb, ub = model.compute_bounds(x=(image,), method=m)
        lb[0][orig_label] = 0
        gap_gx = ub[0][orig_label]-torch.max(lb)
        gap_gx.backward(retain_graph=True)
        time_start = time.time()
        for i in range(args.interval+1):
            # model.zero_grad()
            if i == 0:
                eps = 0.0001
                r_0 = cal_r(image, eps)
                if inputs.grad is not None:
                    inputs.grad.data.zero_()
                inputs_new = inputs.cpu().data.numpy() + r_0.cpu().data.numpy()
                inputs_new = torch.from_numpy(inputs_new).to(device)
                inputs_new.requires_grad_()
                ptb = PerturbationLpNorm(norm=2, eps=eps)
                image_new = BoundedTensor(inputs_new, ptb)

                lb, ub = model.compute_bounds(x=(image_new,), method=m)
                lb[0][orig_label] = 0;
                gap_gx = ub[0][orig_label] - torch.max(lb)
                gap_gx.backward(retain_graph=True)
            else:
                eps_new = (tao) / args.interval
                r = cal_r(image_new, eps_new)
                if inputs_new.grad is not None:
                    inputs_new.grad.data.zero_()
                inputs_new = inputs_new.cpu().data.numpy() + r.cpu().data.numpy()
                inputs_new = torch.from_numpy(inputs_new).to(device)
                r_erro = i * eps_new - torch.norm(inputs_new - inputs)
                num = 0
                while r_erro != 0 and num <= 10:
                    num += 1
                    r_erro = i * eps_new - torch.norm(inputs_new - inputs)
                    r_erro_tensor = cal_r(image_new, r_erro)
                    inputs_new = inputs_new.cpu().data.numpy() + r_erro_tensor.cpu().data.numpy()
                    inputs_new = torch.from_numpy(inputs_new).to(device)
                    r_erro = i * eps_new - torch.norm(inputs_new - inputs)
                    # print(num)
                inputs_new.requires_grad_()
                ptb = PerturbationLpNorm(norm=2, eps=eps_new)
                image_new = BoundedTensor(inputs_new, ptb)

                lb, ub = model.compute_bounds(x=(image_new,), method=m)
                lb[0][orig_label] = 0
                gap_gx = ub[0][orig_label] - torch.max(lb)
                gap_gx.backward(retain_graph=True)

            outputs = model(inputs_new)
            _, predicted = torch.max(outputs.data, 1)
            current_label = predicted.item()
            if current_label != orig_label:
                is_adv = True
                diff = torch.norm(inputs_new - inputs).data
                print("Attack successed!")
                sys.stdout.flush()
                print('r=', diff, 'label is', current_label, 'prob=', _)
                sys.stdout.flush()
            else:
                is_adv = False
                diff = 0
                print('label is', current_label, 'prob=', _)
        if is_adv == True :
            count += 1
        print('##########################################################')
        sys.stdout.flush()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='compute minimum r for CIFAR and MNIST')
    parser.add_argument('--model',
                        default='mnist',
                        choices=['mnist','cifar','tinyimagenet','imagenet'],
                        help='model to be used')
    parser.add_argument('--interval',
                        default=50,
                        type=int,
                        help='we split the eps total into interval parts and the larger the intervals is the smaller the eps upadate each step'
                        )
    # parser.add_argument('--epstotal',
    #                    default=3,
    #                    type= int,
    #                    help='the eps total of added to the sample')
    parser.add_argument('--imagestart',
                        default=0,
                        type=int,
                        help='the start index of iamges needs to calculate')
    parser.add_argument('--imagetotal',
                        default=1,
                        type=int,
                        help='the total of iamges needs to calculate')
    parser.add_argument('--tao',
                        default=0.494,
                        type=float,
                        help='the tao of add perturbations')
    parser.add_argument('--model_name',
                        # default='cifar7_1024',
                        choices=['mnist', 'cifar7_1024', 'tinyimagenet','wide_resnet_cifar_bn_wo_pooling_dropout','ResNeXt_cifar','Densenet_cifar','VGG11','googlenet','resnet18'],
                        default='mnist',
                        type=str,
                        help='the tao of add perturbations')
    # parser.add_argument("--data_dir", type=str, default="data/tinyImageNet/tiny-imagenet-200",
    #                     help='dir of dataset')

    args = parser.parse_args()
    print(args)
    # ['IBP', 'IBP+backward (CROWN-IBP)', 'backward (CROWN)', 'CROWN-Optimized (alpha-CROWN)']:
    m = 'IBP+backward'
    model_type = args.model
    cal_r_bound(model_type,m)
