import os
import torch
from collections import OrderedDict
from abc import ABC, abstractmethod

class BaseModel(ABC):
    """This class is an abstract base class (ABC) for models.
    To create a subclass, you need to implement the following five functions:
        -- <__init__>:                      initialize the class; first call
                                            BaseModel.__init__(self, opt).
        -- <set_input>:                     unpack data from dataset and
                                            apply preprocessing.
        -- <forward>:                       produce intermediate results.
        -- <optimize_parameters>:           calculate losses, gradients,
                                            and update network weights.
        -- <modify_commandline_options>:    (optionally) add model-specific
                                            options and set default options.
    """

    def __init__(self, opt):
        """Initialize the BaseModel class.

        Parameters:
            opt (Option class)-- stores all the experiment flags;
                                 needs to be a subclass of BaseOptions

        When creating your custom class, you need to implement your own
        initialization.
        In this fucntion, you should first call <BaseModel.__init__(self, opt)>
        Then, you need to define four lists:
            -- self.loss_names (str list):          specify the training losses
                                                    that you want to plot and
                                                    save.
            -- self.model_names (str list):         specify the images that you
                                                    want to display and save.
            -- self.visual_names (str list):        define networks used in our
                                                    training.
            -- self.optimizers (optimizer list):    define and initialize
                                                    optimizers. 
        """
        self.opt = opt
        self.gpu_ids = opt.gpu_ids
        self.device = torch.device('cuda:{}'.format(self.gpu_ids[0]))\
                        if self.gpu_ids else torch.device('cpu')
                        # get device name: CPU or GPU

    @abstractmethod
    def get_net_and_loss(self):
        """Get the network layers and losses"""
        pass
    
    @abstractmethod
    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters>
            and <test>."""
        pass

    @abstractmethod
    def optimize_parameters(self):
        """Calculate losses, gradients, and update network weights;
            called in every training iteration"""
        pass

