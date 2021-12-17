import sys
import argparse


class Args(object):
    @staticmethod
    def get_num_class(dataset):
        num = {
            'cifar10': 10,
            'cifar100': 100
        }
        return num[dataset]

    @staticmethod
    def get_img_size(dataset):
        size = {
            'cifar10': 32,
            'cifar100':32
        }
        return size[dataset]


devices = ['cpu', 'cuda']
datasets = ['cifar10', 'cifar100']
models = ['resnet32', 'mobilenetv2_x0_5', 'vgg13_bn', 'shufflenetv2_x1_0']
noises = ['gaussion']
fsmethods = ['featswap', 'perfloss', 'ratioestim']
crtmethods = ['patch', 'finetune', 'sensei']


commparser = argparse.ArgumentParser(add_help=False)
commparser.add_argument('--data_dir', default='data')
commparser.add_argument('--output_dir', default='output')
commparser.add_argument('--device', default='cuda', choices=devices)
commparser.add_argument('--gpu', type=int, default=3, choices=[0, 1, 2, 3])
commparser.add_argument('-b', '--batch_size', type=int, default=256)
commparser.add_argument('-m', '--model', type=str, required=True, choices=models)
commparser.add_argument('-r', '--resume', action='store_true')
data_group = commparser.add_argument_group('dataset')
data_group.add_argument('-d', '--dataset', type=str, required=True, choices=datasets)
data_group.add_argument('-n', '--noise_type', type=str, default='gaussion', choices=noises)
optim_group = commparser.add_argument_group('optimizer')
optim_group.add_argument('--lr', type=float, default=0.01, help='learning rate')
optim_group.add_argument('--momentum', type=float, default=0.9)
optim_group.add_argument('--weight_decay', type=float, default=5e-4)
optim_group.add_argument('-e', '--max_epoch', type=int, default=50)

advparser = argparse.ArgumentParser(parents=[commparser])
advparser.add_argument('-c', '--crt_method', type=str, required=True, choices=crtmethods)
advparser.add_argument('-f', '--fs_method', type=str, default=None, required='patch' in sys.argv, choices=fsmethods)
advparser.add_argument('--crt_type', type=str, choices=['crtunit', 'replace'])
advparser.add_argument('--crt_epoch', type=int, default=20)
advparser.add_argument('--susp_ratio', type=float, default=0.25)
advparser.add_argument('--susp_side', type=str, default='front', choices=['front', 'rear', 'random'])
advparser.add_argument('--robust_threshold', type=float, default=1e-3)
advparser.add_argument('--popsize', type=int, default=10)
advparser.add_argument('--crossover_prob', type=float, default=0.4)

