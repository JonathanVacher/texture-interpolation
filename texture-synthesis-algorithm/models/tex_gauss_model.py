import torch
import numpy as np
import torch.optim as optim

import copy
from .base_model import BaseModel
from . import utils

class TexGaussModel(BaseModel):
    
    def __init__(self, opt):
        """Initialize the TexInterp class.
        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a
                                 subclass of BaseOptions
        """
        BaseModel.__init__(self, opt)
        torch.manual_seed(opt.seed)
        self.input_tex_1 =   utils.image_loader(opt.input_tex_1, opt.gray,
                                                self.device)
        self.input_tex_2 =   utils.image_loader(opt.input_tex_2, opt.gray,
                                                self.device)
        shape = np.fromstring(opt.shape, dtype=int, sep=',')
        self.output_tex = 0.5+0.02*torch.randn((1,1,shape[0],shape[1]),
                                          device=self.device).repeat(1,3,1,1)
        self.output_tex.requires_grad=True
        self.weight_1 = opt.weight_1
        self.lap_reg = opt.lap_reg
        scales = np.fromstring(opt.scales, dtype=int, sep=',')
        self.cnn, self.tex_layers = utils.network_loader(
                                            opt.cnn, not(opt.rdm_weights),
                                            opt.n_layers, opt.weights_dist,
                                            self.device,
                                            opt.n_features, scales)
        self.normalize_mean = torch.tensor([0.485, 0.456, 0.406],
                                           device=self.device)
        self.normalize_std = torch.tensor([0.229, 0.224, 0.225],
                                           device=self.device)
        self.optimizer = optim.LBFGS([self.output_tex],
                                     lr=opt.lr, max_iter=opt.bfgs_iter)
        
    def get_net_and_loss(self):
        """Get the network layers and losses"""
        # normalization module
        normalization = utils.Normalization(self.normalize_mean,
                                            self.normalize_std).to(self.device)

        # just in order to have an iterable access to or list of tex
        # losses
        self.tex_losses_1 = []
        self.tex_losses_2 = []

        # assuming that cnn is a nn.Sequential, so we make a new nn.Sequential
        # to put in modules that are supposed to be activated sequentially
        self.model = torch.nn.Sequential(normalization)
        i = 0  # increment every time we see a conv
        for layer in self.cnn.children():
            if isinstance(layer, torch.nn.Conv2d):
                i += 1
                name = 'conv_{}'.format(i)
            elif isinstance(layer, torch.nn.ModuleDict):
                i += 1
                name = 'conv_list_{}'.format(i)
            elif isinstance(layer, torch.nn.ReLU):
                name = 'relu_{}'.format(i)
                # The in-place version doesn't play very nicely with the TexLoss
                # we insert below. So we replace with out-of-place ones here. 
                layer = torch.nn.ReLU(inplace=False)
            elif isinstance(layer, torch.nn.MaxPool2d):
                name = 'pool_{}'.format(i)
            elif isinstance(layer, torch.nn.BatchNorm2d):
                name = 'bn_{}'.format(i)
            else:
                raise RuntimeError('Unrecognized layer: {}'
                                   .format(layer.__class__.__name__))

            self.model.add_module(name, layer)

            if name in self.tex_layers:
                # add tex loss:
                target_feature_1 = self.model(self.input_tex_1).detach()
                target_feature_2 = self.model(self.input_tex_2).detach()
                
                tex_loss_1 = utils.TexLossMeanCov(target_feature_1, self.device)
                tex_loss_2 = utils.TexLossMeanCov(target_feature_2, self.device)
                
                self.model.add_module("tex_loss_1_{}".format(i), tex_loss_1)
                self.model.add_module("tex_loss_2_{}".format(i), tex_loss_2)
                
                self.tex_losses_1.append(tex_loss_1)
                self.tex_losses_2.append(tex_loss_2)

        # now we trim off the layers after the last tex losses
        for i in range(len(self.model) - 1, -1, -1):
            if isinstance(self.model[i], utils.TexLossMeanCov):
                break

        self.model = self.model[:(i + 1)]
        
        #self.model = model#.to(self.device)
        #self.tex_losses = tex_losses
        #return model, tex_losses
    
    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> 
                             and <test>."""
        self.model(self.output_tex)
        
        
    def optimize_parameters(self):
        """Calculate losses, gradients, and update network weights; called
            in every training iteration"""
        def closure():
            # correct the values of updated input image
            self.output_tex.data.clamp_(0, 1)
            self.optimizer.zero_grad()
            self.forward()
            self.tex_score_1 = 0
            self.tex_score_2 = 0
            for i in range(len(self.tex_losses_1)):
                self.tex_score_1 += self.tex_losses_1[i].loss
                self.tex_score_2 += self.tex_losses_2[i].loss
                
            self.tex_score = self.weight_1*self.tex_score_1\
                            +(1-self.weight_1)*self.tex_score_2
            self.tex_score += self.lap_reg*utils.norm_laplacian(self.output_tex)
            
            loss = self.tex_score
            loss.backward()
            return self.tex_score
        
        self.optimizer.step(closure)