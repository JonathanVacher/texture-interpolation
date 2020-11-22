"""Deep texture interpolation.

Example:
    Synthesize a texture that interpolate two input textures :
        python3 run_synthesis.py --input_tex_1 curvy.jpg
                                 --input_tex_2 woven.jpg
                                 --weight_1 0.5
    
See options/base_options.py and options/train_options.py for more training options.
"""
import time
import visdom
from options.base_options import BaseOptions
from models import create_model, utils
import torch


if __name__ == '__main__':
    opt = BaseOptions().parse()   # get training options
    
    if utils.socket_is_used(opt.display_port,'localhost'):
        vis = visdom.Visdom(port=opt.display_port, use_incoming_socket=False)
    else: 
        print('Visom server is not running at localhost:'+str(opt.display_port))
        
    model = create_model(opt) # create a model given opt.model and other options
    
    model.get_net_and_loss()
    init_time = time.time()        
    for epoch in range(opt.n_iter+1):    
        model.optimize_parameters()
        if epoch%opt.display_freq==0:
            image = 255*model.output_tex.data.clamp_(0,1).squeeze(0)
            if utils.socket_is_used(opt.display_port,'localhost'):
                vis.image(image, win=opt.win)
            print("Iteration : {}".format(epoch))
            print('Duration : {}'.format(time.time()-init_time))
            print('Tex Loss : {:4f}'.format(model.tex_score.item()))
    
    output_tex = model.output_tex.data.clamp_(0,1) 
    
    utils.save_image(utils.tensor2im(output_tex), opt.output_tex,
                     aspect_ratio=1.0)
                     
