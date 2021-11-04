import argparse
import os
from dataset import get_loader
from solver import Solver
import glob
import shutil


def main(config):
    if config.mode == 'train':
        train_loader, dataset = get_loader(config.batch_size, num_thread=config.num_thread)
        if not os.path.exists(os.path.join(config.save_fold, config.modelname)):
            os.makedirs(os.path.join(config.save_fold, config.modelname))
            os.makedirs(os.path.join(config.save_fold, config.modelname, 'code'))
            os.makedirs(os.path.join(config.save_fold, config.modelname, 'models'))
            os.makedirs(os.path.join(config.save_fold, config.modelname, 'tmp_salmap'))
            cwd = os.getcwd()
            for ff in glob.glob('*.py'):
                shutil.copy2(os.path.join(cwd, ff), config.save_fold + config.modelname + '/code/')
        train = Solver(train_loader, None, config)
        train.train()
    elif config.mode == 'test':
        test_loader, dataset = get_loader(config.test_batch_size, mode='test', num_thread=config.num_thread, test_mode=config.test_mode, sal_mode=config.sal_mode)
        test = Solver(None, test_loader, config, dataset.save_folder())
        test.test(test_mode=config.test_mode)
    else:
        raise IOError("illegal input")


if __name__ == '__main__':

    vgg_path = 'vgg16_20M.pth'
    resnet_path = 'resnet50_caffe.pth'

    parser = argparse.ArgumentParser()

    parser.add_argument('--cuda', type=bool, default=True)

    # Training settings
    parser.add_argument('--vgg', type=str, default=vgg_path) #'transfer learning, load 16 layers of pretrained vgg', load in build_model
    parser.add_argument('--resnet', type=str, default=resnet_path)
    parser.add_argument('--epoch', type=int, default=50)  #VGG-40  ResNet-30
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--test_batch_size', type=int, default=1)
    parser.add_argument('--num_thread', type=int, default=4)
    parser.add_argument('--load_bone', type=str, default='')
    # parser.add_argument('--load_branch', type=str, default='')
    parser.add_argument('--save_fold', type=str, default='/db/psxbd1/logs/')
    parser.add_argument('--pre_trained', type=str, default=None)
    #parser.add_argument('--validation', type=int, default=1)
    parser.add_argument('--modelname', type=str, default='ResNet')
    # Testing settings
    parser.add_argument('--model', type=str, default='/db/psxbd1/DSLRD-ResNet.pth') #location of the trained final model in test model
    #parser.add_argument('--test_fold', type=str, default='results/test/')
    parser.add_argument('--test_mode', type=int, default=1)
    parser.add_argument('--sal_mode', type=str, default='m')

    # Misc
    parser.add_argument('--mode', type=str, default='test', choices=['train', 'test'])
    parser.add_argument('--visdom', type=bool, default=False)

    config = parser.parse_args()

    if not os.path.exists(config.save_fold):
        os.mkdir(config.save_fold)
    main(config)