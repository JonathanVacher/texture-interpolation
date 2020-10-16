import argparse
import os
from models import utils
import torch
import models


class BaseOptions():
    """This class defines options used during both training and test time.

    It also implements several helper functions such as parsing, printing,
    and saving the options.
    It also gathers additional options defined in <modify_commandline_options>
    functions in both dataset class and model class.
    """

    def __init__(self):
        """Reset the class; indicates the class hasn't been initailized"""
        self.initialized = False

    def initialize(self, parser):
        """Define the common options that are used in both training and test."""
        # model parameters
        parser.add_argument('--weight_1', type=float, default=0.5,\
                            help='weight of input_tex_1 (between 0 and 1)')
        parser.add_argument('--input_tex_1', type=str, default='woven.jpg',\
                            help='# of input image channels:'\
                                 +' 3 for RGB and 1 for grayscale')
        parser.add_argument('--input_tex_2', type=str, default='curvy.jpg',\
                            help='# of input image channels:'\
                                 +' 3 for RGB and 1 for grayscale')
        parser.add_argument('--output_tex', type=str, default='output_tex.png',\
                            help='# of output image channels:'\
                                 +' 3 for RGB and 1 for grayscale')
        parser.add_argument('--gray', action='store_true',\
                            help='convert rgb input to gray')
        parser.add_argument('--lap_reg', type=float, default=0.0,\
                            help='laplacian regularization parameter')
        parser.add_argument('--spectrum_constraint', type=float, default=0.0,\
                            help='spectrum constraint weight,'\
                                +' prefered 0.0 or 1.0, only with tex_wasser'
                                +' model, useful to guaranty power spectrum')
        parser.add_argument('--model', type=str, default='tex_wasser',\
                            help='choose which model to use')
        parser.add_argument('--cnn', type=str, default='vgg19',\
                            help='choose which cnn to use '\
                                + 'vgg19/alexnet/multiscale')
        parser.add_argument('--interp_method', type=str, default='1',\
                            help='choose interpolation method for the'\
                                 +' Wasserstein loss')
        parser.add_argument('--n_features', type=int, default=128,\
                            help='number of features when cnn is "multiscale"')
        parser.add_argument('--scales', type=str,\
                            default='3,5,7,11,15,23,37,55',\
                            help='list of scales when cnn is "multiscale"')
        parser.add_argument('--rdm_weights', action='store_true',\
                            help='choose pretrained weights or not')
        parser.add_argument('--weights_dist', type=str, default='unif',\
                            help='choose weights dist gauss/unif')
        parser.add_argument('--n_layers', type=int, default=5,\
                            help='choose the number of conv layer to use')
        # basic parameters
        parser.add_argument('--gpu_ids', type=str, default='0',\
                            help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
        
        # visdom and HTML visualization parameters
        parser.add_argument('--display_freq', type=int, default=50,\
                            help='frequency of showing'\
                                 +' training results on screen')
        parser.add_argument('--display_port', type=int,\
                            default=8892, help='visdom port of the web display')
        parser.add_argument('--win', type=int,\
                            default=2, help='visdom window number')
        # training parameters
        parser.add_argument('--bfgs_iter', type=int, default=20,\
                            help='# of iter inside 1 bfgs step')
        parser.add_argument('--n_iter', type=int, default=150,
                            help='# of iter ')
        parser.add_argument('--lr', type=float, default=0.1,
                            help='learning rate')
        parser.add_argument('--seed', type=int, default=0,
                            help='random seed')
        parser.add_argument('--shape', type=str, default='256,256',\
                            help='shape of the output texture')
        parser.add_argument('--verbose', action='store_true',\
                            help='if specified, print'
                                 +' more debugging information')
        
        return parser

    def gather_options(self):
        """Initialize our parser with basic options(only once).
        Add additional model-specific and dataset-specific options.
        These options are defined in the <modify_commandline_options> function
        in model and dataset classes.
        """
        if not self.initialized:  # check if it has been initialized
            formatter = lambda prog: argparse.HelpFormatter(
                                                prog,max_help_position=52)
            parser = argparse.ArgumentParser(formatter_class=formatter)
            parser = self.initialize(parser)
            
        # get the basic options
        opt, _ = parser.parse_known_args()

        # save and return the parser
        self.parser = parser
        return parser.parse_args()

    def print_options(self, opt):
        """Print and save options

        It will print both current options and default values(if different).
        It will save options into a text file / [checkpoints_dir] / opt.txt
        """
        message = ''
        message += '----------------- Options ---------------\n'
        for k, v in sorted(vars(opt).items()):
            comment = ''
            default = self.parser.get_default(k)
            if v != default:
                comment = '\t[default: %s]' % str(default)
            message += '{:>25}: {:<30}{}\n'.format(str(k), str(v), comment)
        message += '----------------- End -------------------'
        print(message)

    def parse(self):
        """Parse our options, create checkpoints directory suffix,
            and set up gpu device."""
        opt = self.gather_options()

        self.print_options(opt)

        # set gpu ids
        str_ids = opt.gpu_ids.split(',')
        opt.gpu_ids = []
        for str_id in str_ids:
            id = int(str_id)
            if id >= 0:
                opt.gpu_ids.append(id)
        if len(opt.gpu_ids) > 0:
            torch.cuda.set_device(opt.gpu_ids[0])

        self.opt = opt
        return self.opt
