import torch
import torch.fft
from torch import nn
from torch.autograd import Function
import torchvision.transforms as transforms
import torchvision.models as models
import numpy as np
from PIL import Image
import os
import socket


""" Utilitary functions
- socket_is_used
- init_weights_unif
- init_weights_gauss
- init_weights_unif_smoothed
- init_weights_gauss_smoothed
- model_multiscale
- model_multiscale1
- network_loader
- image_loader
- tensor2im
- save_image
- mkdirs
- mkdir
- histogram_matching_1d
- sliced_wasserstein_3d
- norm_laplacian
- gram_matrix
- load_network
- Normalization
- symsqrt
- MatrixSquareRoot
- wasserstein_spd_loss
- kl_div
- TexLossGram
- TexLossMeanCov
- TexLossKL
- TexLossOptTrans
- TexLossOptTransAlternative
- fft2
- ifft2
- power_spectrum
- power_spectrum_constraint
"""

def socket_is_used(port, hostname):
    is_used = False
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        s.bind((hostname, port))
    except socket.error:
        is_used = True
    finally:
        s.close()
    return is_used

def init_weights_unif(layer):
    """ Initializing rdm weights with uniform distribution
    
    Parameters:
        layer (Conv2d layer) -- a pytorch conv2 layer
    """
    if type(layer) == torch.nn.Conv2d:
        torch.nn.init.xavier_uniform_(layer.weight,3.)
        layer.bias.data.fill_(0.0)

def init_weights_gauss(layer):
    """ Initializing rdm weights with gaussian distribution
    
    Parameters:
        layer (Conv2d layer) -- a pytorch conv2 layer
    """
    if type(layer) == torch.nn.Conv2d:
        torch.nn.init.xavier_normal_(layer.weight,3.)
        layer.bias.data.fill_(0.0)
        
def init_weights_unif_smoothed(layer):
    """ Initializing rdm weights with uniform distribution and smooth
    
    Parameters:
        layer (Conv2d layer) -- a pytorch conv2 layer
    """
    if type(layer) == torch.nn.Conv2d:
        n = layer.in_channels
        p = layer.kernel_size[0]
        half_p = (p%2)*(p-1)//2 + ((p+1)%2)*p//2
        half_n = (n%2)*(n-1)//2 + ((n+1)%2)*n//2
        smooth = torch.nn.Conv2d(n, n,
                    kernel_size=(p,p),
                    stride=(1, 1),
                    padding=(half_p, half_p)).to(layer.weight.device)
        smooth.bias.data.fill_(0.0)
        x = torch.arange(-half_p,half_p+p%2).type(layer.weight.dtype)
        z = torch.arange(-half_n,half_n+n%2).type(layer.weight.dtype)
        z = torch.roll(z, half_n+n%2, 0)
        Z, X, Y = torch.meshgrid(z,x,x)
        sig_xy = 2.0
        sig_z = 1.0
        kernel = torch.exp(-(X**2+Y**2)/(2*sig_xy**2)-Z**2/(2*sig_z**2))
        kernel /= kernel.norm()
        for i in range(n):
            smooth.weight.data[i]= torch.roll(kernel,i,0)
        
        torch.nn.init.xavier_uniform_(layer.weight,1.)
        layer.weight.data = smooth(layer.weight.data)
        layer.bias.data.fill_(0.0)

def init_weights_gauss_smoothed(layer):
    """ Initializing rdm weights with gaussian distribution and smooth
    
    Parameters:
        layer (Conv2d layer) -- a pytorch conv2 layer
    """
    if type(layer) == torch.nn.Conv2d:
        n = layer.in_channels
        p = layer.kernel_size[0]
        half_p = (p%2)*(p-1)//2 + ((p+1)%2)*p//2
        half_n = (n%2)*(n-1)//2 + ((n+1)%2)*n//2
        smooth = torch.nn.Conv2d(n, n,
                    kernel_size=(p,p),
                    stride=(1, 1),
                    padding=(half_p, half_p)).to(layer.weight.device)
        smooth.bias.data.fill_(0.0)
        x = torch.arange(-half_p,half_p+p%2).type(layer.weight.dtype)
        z = torch.arange(-half_n,half_n+n%2).type(layer.weight.dtype)
        z = torch.roll(z, half_n+n%2, 0)
        Z, X, Y = torch.meshgrid(z,x,x)
        sig_xy = 2.0
        sig_z = 1.0
        kernel = torch.exp(-(X**2+Y**2)/(2*sig_xy**2)-Z**2/(2*sig_z**2))
        kernel /= kernel.norm()
        for i in range(n):
            smooth.weight.data[i]= torch.roll(kernel,i,0)
        
        torch.nn.init.xavier_normal_(layer.weight,1.)
        layer.weight.data = smooth(layer.weight.data)
        layer.bias.data.fill_(0.0)

class model_multiscale1(nn.ModuleDict):
    """ Helper to generate a multiscale model
    """
    def forward(self, input):
        out_ = [module(input) 
                for _, module in self.items()]
        out = torch.cat(out_, 1)
        return nn.functional.relu(5*out)

class model_multiscale(nn.ModuleDict):
    """ Helper to generate a multiscale model
    """
    def forward(self, input):
        out_ = [module(input) 
                for _, module in self.items()]
        out = torch.cat(out_, 1)
        return out
        
def network_loader(name, pretrained, n_layers, distribution, device,
                   n_features=128, scales=[3, 5, 7, 11, 15, 23, 37, 55]):
    """ Load any of the following networks: vgg19, alexnet, mutliscale1/2/3
    
    Parameters:
        name (str)           -- network id
        pretrained (boolean) -- load pretrained weights (only vgg19 and alexnet)
        n_layers (int)       -- number of layers for texture constraints
        distribution (str)   -- distribution of rdm weights (not pretrained)
        device (int)         -- device on which to load the network
        n_features (int)     -- number of neurons (multiscale only)
        scales (list int)    -- filter sizes (length must divide n_features)
    
    """
    if name=='vgg19':
        model = models.vgg19(pretrained=pretrained).features.to(device)
        tex_layers = ['conv_'+str(i) for i in range(1,n_layers+1)]
    elif name=='alexnet':
        model = models.alexnet(pretrained=pretrained).features.to(device)
        tex_layers = ['conv_'+str(i) for i in range(1,n_layers+1)]
    elif name=='multiscale':
        model = nn.Sequential()
        model.add_module('conv_list_0',
                         model_multiscale1(
                            {'conv_0%i'%(i):nn.Conv2d(3, n_features, 
                                kernel_size=scales[i],
                                padding=scales[i]//2) 
                                for i in range(len(scales))}))
        model.to(device)
        tex_layers = ['conv_list_1']
        pretrained = False
    elif name=='multiscale2':
        model = nn.Sequential()
        model.add_module('conv_list_0',
                         model_multiscale(
                            {'conv_0%i'%(i):nn.Conv2d(3, n_features, 
                                kernel_size=scales[i],
                                padding=scales[i]//2) 
                                for i in range(len(scales))}))
        model.add_module('relu_0', nn.ReLU())
        model.add_module('pool_0', nn.MaxPool2d(kernel_size=2))
        model.add_module('conv_list_1',
                         model_multiscale(
                            {'conv_0%i'%(i):nn.Conv2d(n_features*len(scales),
                                n_features, 
                                kernel_size=scales[i],
                                padding=scales[i]//2) 
                                for i in range(len(scales))}))
        
        model.to(device)
        tex_layers = ['conv_list_1','conv_list_2']
        pretrained = False
    elif name=='multiscale3':
        model = nn.Sequential()
        model.add_module('conv_list_0',
                         model_multiscale(
                            {'conv_0%i'%(i):nn.Conv2d(3, n_features, 
                                kernel_size=scales[i],
                                padding=scales[i]//2) 
                                for i in range(len(scales))}))
        model.add_module('relu_0', nn.ReLU())
        model.add_module('pool_0', nn.MaxPool2d(kernel_size=2))
        model.add_module('conv_list_1',
                         model_multiscale(
                            {'conv_0%i'%(i):nn.Conv2d(n_features*len(scales),
                                n_features, 
                                kernel_size=scales[i],
                                padding=scales[i]//2) 
                                for i in range(len(scales))}))
        model.add_module('relu_1', nn.ReLU())
        model.add_module('pool_1', nn.MaxPool2d(kernel_size=2))
        model.add_module('conv_list_2',
                         model_multiscale(
                            {'conv_0%i'%(i):nn.Conv2d(n_features*len(scales),
                                n_features, 
                                kernel_size=scales[i],
                                padding=scales[i]//2) 
                                for i in range(len(scales))}))
        
        model.to(device)
        tex_layers = ['conv_list_1','conv_list_2','conv_list_3']
        pretrained = False
    
        
    if not pretrained:
        if distribution=='unif':
            model.apply(init_weights_unif)
        elif distribution=='unif_s':
            model.apply(init_weights_unif_smoothed)
        elif distribution=='gauss':
            model.apply(init_weights_gauss)
        elif distribution=='gauss_s':
            model.apply(init_weights_gauss_smoothed)
            
    return model, tex_layers
        
def image_loader(image_name, gray, device):
    """ Load image on device in color or gray
    
    Parameters:
        image_name (str) -- the image file name
        gray (boolean)      -- load image in grayscale
        device (int)        -- device on which to send the image
    """
    if gray:
        image = Image.open(image_name).convert('L')
    else:
        image = Image.open(image_name)
    # fake batch dimension required to fit network's input dimensions
    image = transforms.ToTensor()(image).unsqueeze(0)
    return image.to(device, torch.float)

def tensor2im(input_image, imtype=np.uint8):
    """Converts a Tensor array into a numpy image array.

    Parameters:
        input_image (tensor) --  the input image tensor array
        imtype (type)        --  the desired type of the converted numpy array
    """
    if not isinstance(input_image, np.ndarray):
        if isinstance(input_image, torch.Tensor):  # get the data from a variable
            image_tensor = input_image.data
        else:
            return input_image
        image_numpy = image_tensor[0].cpu().float().numpy()  # convert it into a numpy array
        if image_numpy.shape[0] == 1:  # grayscale to RGB
            image_numpy = np.tile(image_numpy, (3, 1, 1))
        image_numpy = np.transpose(image_numpy, (1, 2, 0)) * 255.0  # post-processing: tranpose and scaling
    else:  # if it is a numpy array, do nothing
        image_numpy = input_image
    return image_numpy.astype(imtype)

def save_image(image_numpy, image_path, aspect_ratio=1.0):
    """Save a numpy image to the disk

    Parameters:
        image_numpy (numpy array) -- input numpy array
        image_path (str)          -- the path of the image
    """

    image_pil = Image.fromarray(image_numpy)
    h, w, _ = image_numpy.shape

    if aspect_ratio > 1.0:
        image_pil = image_pil.resize((h, int(w * aspect_ratio)), Image.BICUBIC)
    if aspect_ratio < 1.0:
        image_pil = image_pil.resize((int(h / aspect_ratio), w), Image.BICUBIC)
    image_pil.save(image_path)

def mkdirs(paths):
    """create empty directories if they don't exist

    Parameters:
        paths (str list) -- a list of directory paths
    """
    if isinstance(paths, list) and not isinstance(paths, str):
        for path in paths:
            mkdir(path)
    else:
        mkdir(paths)

def mkdir(path):
    """create a single empty directory if it didn't exist

    Parameters:
        path (str) -- a single directory path
    """
    if not os.path.exists(path):
        os.makedirs(path)


def histogram_matching_1d(input_image, target_image):
    """
    Matches histogram of a single channel image
    
    Parameters:
    input_image (tensor)  -- image whose distribution should be remapped
    target_image (tensor) -- image whose distribution should be matched
    
    Output:
    ouput_image (tensor) -- an image with same histogram as target_image
    """
    ouput_image = torch.zeros_like(input_image).view(-1)
    input_image_sorted = torch.argsort(input_image.view(-1))
    ouput_image[input_image_sorted], _ = torch.sort(target_image.view(-1))
    ouput_image = ouput_image.view(input_image.size())
    
    return ouput_image

def sliced_wasserstein_3d(input_image, target_image_1,
                               target_image_2, weight, n_iter=250):
    """
    Interpolate between two 3d clouds (color historgrams) using sliced 
    wasserstein approximation
    
    Parameters:
    input_image (tensor)    -- image whose distribution should be remapped
    target_image_1 (tensor) -- image whose distribution should be matched
    target_image_2 (tensor) -- image whose distribution should be matched
    weight (float)          -- weight of target_image_1
    
    Output:
    ouput_image (tensor) -- an image with interpolated distribution
    """
    
    _, Nc, Ny, Nx = input_image.shape
    _, _, Ny_, Nx_ = target_image_1.shape
    output_image = input_image.clone().view(Nc,Ny*Nx)
    target_1 = target_image_1.view(Nc,Ny_*Nx_)
    target_2 = target_image_2.view(Nc,Ny_*Nx_)
    tau = 0.2
    idx = torch.arange(Nx*Ny).long()
    
    for i in range(n_iter):   
        if Ny!=Ny_:
            perm = torch.randperm(Nx*Ny)
            idx = perm[:Nx_*Ny_]
        
        Theta, _ = torch.qr(torch.randn(Nc, Nc, dtype=input_image.dtype,
                                        device=input_image.device)) 
        P1 = histogram_matching_1d(torch.mm(Theta.t(),output_image[:,idx]),
                                torch.mm(Theta.t(),target_1))
        P2 = histogram_matching_1d(torch.mm(Theta.t(),output_image[:,idx]),
                                torch.mm(Theta.t(),target_2))
        Delta = weight*P1+(1-weight)*P2
        output_image[:,idx] = (1-tau)*output_image[:,idx]\
                             + tau*torch.mm(Theta,Delta)
    
    output_image = output_image.view(1,Nc,Ny,Nx)
    return output_image

def norm_laplacian(input):
    """ Compute the norm of the 2d-laplacian of input
    
    Parameters:
        input (float tensor) -- an image array
    """
    b, c, h, w = input.shape
    kernel = torch.zeros(c, c, 3, 3, device=input.device, dtype=input.dtype)

    for i in range(3):
        kernel[i,i] = -torch.ones(3, 3, device=input.device, dtype=input.dtype)
        kernel[i,i,1,1] = 8.0
    laplacian_input = torch.nn.functional.conv2d(input, kernel, 
                                                 padding=(1,1))
    norm_laplacian_input = torch.norm(laplacian_input, p=1)**1
    norm_laplacian_input /= h*w
    return norm_laplacian_input

def gram_matrix(input, device):
    """ Compute gram matrix of input
    
    Parameters:
        input (float tensor) -- rectangular array
        device (int)         -- device of input
    """
    a, b = input.size()  
    
    G = torch.mm(input, input.t())+1e-6*torch.eye(a, device=device)  
    # compute the gram product
    # we 'normalize' the values of the gram matrix
    # by dividing by the number of element in each feature maps.
    # This computation corresponds to the covariance matrix of the features.
    return G.div(b)

# create a module to normalize input image so we can easily put it in a
# nn.Sequential, this is not essential
class Normalization(torch.nn.Module):
    """ Normalization layer """
    def __init__(self, mean, std):
        super(Normalization, self).__init__()
        self.mean = mean.clone().detach().view(-1, 1, 1)
        self.std = std.clone().detach().view(-1, 1, 1)
        
    def forward(self, img):
        # normalize img
        return (img - self.mean) / self.std

def symsqrt(matrix):
    """Compute the square root of a positive definite matrix.
    [issues with this methods ...]
    """
    
    # perform the decomposition
    s, v = matrix.symeig(eigenvectors=True)
    # truncate small components
    above_cutoff = s > s.max() * s.size(-1) * torch.finfo(s.dtype).eps
    s = s[..., above_cutoff]
    v = v[..., above_cutoff]
    # compose the square root matrix
    return (v * s.sqrt().unsqueeze(-2)) @ v.transpose(-2, -1)

class MatrixSquareRoot(Function):
    """Square root of a positive definite matrix.
    NOTE: matrix square root is not differentiable for matrices with
          zero eigenvalues.
          
    See Lin, Tsung-Yu, and Subhransu Maji. 
        "Improved Bilinear Pooling with CNNs." BMVC 17 
    """    
    @staticmethod
    def forward(ctx, input):
        dim = input.shape[0]
        norm = torch.norm(input.double())
        Y = input/norm
        I = torch.eye(dim,dim,device=input.device).type(input.dtype)
        Z = torch.eye(dim,dim,device=input.device).type(input.dtype)
        for i in range(15):
            T = 0.5*(3.0*I - Z.mm(Y))
            Y = Y.mm(T)
            Z = T.mm(Z)
        sqrtm = Y*torch.sqrt(norm)
        #ctx.mark_dirty(Y,I,Z)
        ctx.save_for_backward(sqrtm)
        return sqrtm#, I, Y, Z

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = None
        sqrtm, = ctx.saved_tensors
        dim = sqrtm.shape[0]
        norm = torch.norm(sqrtm)
        A = sqrtm/norm
        I = torch.eye(dim, dim, device=sqrtm.device).type(sqrtm.dtype)
        Q = grad_output/norm
        for i in range(15):
            Q = 0.5*(Q.mm(3.0*I-A.mm(A))-A.t().mm(A.t().mm(Q)-Q.mm(A)))
            A = 0.5*A.mm(3.0*I-A.mm(A))
        grad_input = 0.5*Q
        return grad_input
    
sqrtm = MatrixSquareRoot.apply

def wasserstein_spd_loss(output, target, sqrt_tar):
    """ Computer the Bures distance between two spd matrix
    
    Parameters:
        output (float tensor)   -- variable matrix
        target (float tensor)   -- objective matrix
        sqrt_tar (float tensor) -- matrix square root of target
    """
    sqrt_out = sqrtm(output)
    C = torch.mm(sqrt_out, sqrt_tar)
    #C = sqrtm(torch.mm(torch.mm(sqrt_tar,output),sqrt_tar))
    loss = torch.trace(output+target)
    loss += -2*torch.trace(C)
    return loss

def kl_div(output_mu, output_sig, target_mu, target_sig, device):
    """ Compute the KL-divergence between 2 Gaussian distributions
    [Lots of numerical issues]
    
    Parameters:
        output_mu (1d tensor)  -- mean of the variables
        output_sig (2d tensor) -- covariance of the variables
        target_mu (1d tensor)  -- target mean
        target_sig (2d tensor) -- target covariance
        device (int)           -- device of the arrays above
    """
    output_sig_inv = torch.inverse(output_sig)
    target_sig_inv = torch.inverse(target_sig)
    
    loss1 = torch.dot(output_mu-target_mu,\
                      torch.mv(output_sig_inv+
                               target_sig_inv,
                               output_mu-target_mu))
    loss2 = torch.trace(output_sig_inv.mm(target_sig_inv))\
           + torch.trace(target_sig_inv.mm(output_sig_inv))
    
    loss = 0.0*loss1 + 1e-24*torch.pow(loss2,2)
    return loss

# Gatys NIPS 2015
# Gram matrix
class TexLossGram(torch.nn.Module):
    """ Constraint layer for the Gram loss"""
    def __init__(self, target_feature, device):
        super(TexLossGram, self).__init__()
        self.device = device
        a, b, c, d = target_feature.size()  
        X = target_feature.view(a*b, c*d)
        self.targetG = gram_matrix(X, self.device)
        
    def forward(self, input):
        a, b, c, d = input.size()
        X = input.view(a*b, c*d)
        G = gram_matrix(X, self.device)
        self.loss = torch.nn.functional.mse_loss(G, self.targetG,
                                                 reduction='sum')
        self.loss /= b**2
        return input
    
# Gatys NIPS 2015
# Here we slightly modify the algorithm to adjust 
# the mean and covariance instead of the gram matrix.
class TexLossMeanCov(torch.nn.Module):
    """ Constraint layer for the mean and covariance loss """
    def __init__(self, target_feature, device):
        super(TexLossMeanCov, self).__init__()
        self.device = device
        a, b, c, d = target_feature.size()  
        X = target_feature.view(a*b, c*d)
        self.targetM = X.mean(1)
        self.targetG = gram_matrix(X-self.targetM.view(a*b, 1), self.device)
        
    def forward(self, input):
        a, b, c, d = input.size()
        X = input.view(a*b, c*d)
        M = X.mean(1)
        G = gram_matrix(X-M.view(a*b, 1), self.device)
        self.loss = torch.nn.functional.mse_loss(G, self.targetG,
                                                   reduction='sum')\
                    + torch.nn.functional.mse_loss(M, self.targetM,
                                                   reduction='sum')
        self.loss /= b**2
        return input

# Kullback Leibler between Gaussians
class TexLossKL(torch.nn.Module):
    """ Constraint layer for KL-divergence loss """
    def __init__(self, target_feature, device):
        super(TexLossKL, self).__init__()
        self.device = device
        a, b, c, d = target_feature.size()  
        X = target_feature.view(a*b, c*d)
        self.targetM = X.mean(1)
        self.targetG = gram_matrix(X-self.targetM.view(a*b, 1),\
                                       self.device)
    def forward(self, input):
        a, b, c, d = input.size()
        X = input.view(a*b, c*d)
        M = X.mean(1)
        G = gram_matrix(X-M.view(a*b, 1), self.device)
        self.loss = kl_div(M, G, self.targetM, self.targetG, self.device)\
                    + torch.nn.functional.mse_loss(M, self.targetM,
                                                   reduction='sum')
        return input

# Here we use the wasserstein distance between 
# gaussian distribution which approx the distance
# between elliptical distribution (Peyre 2019
# Computational optimal transport, remark 2.32)
class TexLossOptTrans(torch.nn.Module):
    """ Constraint layer for the wasserstein loss """
    def __init__(self, target_feature, device):
        super(TexLossOptTrans, self).__init__()
        self.device = device
        a, b, c, d = target_feature.size()  
        X = target_feature.view(a*b, c*d)
        self.targetM = X.mean(1)
        self.targetG = gram_matrix(X-self.targetM.view(a*b, 1), self.device)
        self.sqrtTargetG = sqrtm(self.targetG)
        
    def forward(self, input):
        a, b, c, d = input.size()
        X = input.view(a*b, c*d)
        M = X.mean(1)
        G = gram_matrix(X-M.view(a*b, 1), self.device)
        self.loss = torch.nn.functional.mse_loss(M, self.targetM,
                                                 reduction='sum')\
                    + wasserstein_spd_loss(G, self.targetG, self.sqrtTargetG)
        self.loss /= b
        return input
    
# Alternative wasserstein interpolation
class TexLossOptTransAlternative(torch.nn.Module):
    """ Constraint layer for the wasserstein loss 
     (Alternative using close forms target mean and covariance)
    """
    
    def __init__(self, target_feature_1, target_feature_2, weight, device):
        super(TexLossOptTransAlternative, self).__init__()
        self.device = device
        a, b, c, d = target_feature_1.size()  
        X1 = target_feature_1.view(a*b, c*d)
        X2 = target_feature_2.view(a*b, c*d)
        self.targetM = weight*X1.mean(1)+(1-weight)*X2.mean(1)
        
        C1 = gram_matrix(X1-X1.mean(1, keepdim=True), self.device)
        C2 = gram_matrix(X2-X2.mean(1, keepdim=True), self.device)
        sqrtC1 = sqrtm(C1)
        #sqrtC2 = sqrtm(C2)
        W = sqrtm(torch.mm(torch.mm(sqrtC1,C2),sqrtC1))
        W = torch.mm(torch.inverse(sqrtC1),W)
        self.sqrtTargetG = weight*sqrtC1 + (1-weight)*W
        self.targetG = torch.mm(self.sqrtTargetG,self.sqrtTargetG)
        
    def forward(self, input):
        a, b, c, d = input.size()
        X = input.view(a*b, c*d)
        M = X.mean(1)
        G = gram_matrix(X-M.view(a*b, 1), self.device)
        self.loss = torch.nn.functional.mse_loss(M, self.targetM,
                                                 reduction='sum')\
                    + wasserstein_spd_loss(G, self.targetG, self.sqrtTargetG)
        self.loss /= b
        return input
    
# Helper for batch 2d fft of real number
def fft2(input):
    output = torch.fft.fft(input, 2, norm='ortho')
    return output

# Helper for batch 2d inverse fft with symmetries
# (corresponding to a real signal in spatial domain)
def ifft2(input):
    output = torch.fft.ifft(input, 2, norm='ortho')
    return output.real

# helper for power spectrum computation
def power_spectrum(input):
    output = torch.abs(fft2(input))
    return output

# helper for power spectrum constraint
def power_spectrum_constraint(input_image, target_ps):
    _, _, c, d = target_ps.size()
    output = torch.norm(torch.flatten(power_spectrum(input_image))
                        -torch.flatten(target_ps))**2
    return output/(c*d)

    