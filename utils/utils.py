# coding:utf8
import os
import shutil
import visdom
import time
import torch
import torchvision as tv
import numpy as np
import torchvision.transforms as transforms
from torch.autograd import Variable

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

norm = transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
denorm = transforms.Compose([
    transforms.Normalize(mean=[0, 0, 0], std=[1/0.229, 1/0.224, 1/0.225]),
    transforms.Normalize(mean=[-0.485, -0.456, -0.406], std=[1, 1, 1])
])

class Visualizer():
    """
        A wrapper of visdom, you can still access visdom 
        through `self.vis.function`
    """

    def __init__(self, env='default', **kwargs):
        import visdom
        self.vis = visdom.Visdom(env=env, use_incoming_socket=False, **kwargs)

        self.index = {}
        self.log_text = ''

    def reinit(self, env='default', **kwargs):
        self.vis = visdom.Visdom(env=env,use_incoming_socket=False,  **kwargs)
        return self

    def plot_many(self, d):
        """
        @params d: dict (name,value) i.e. ('loss',0.11)
        """
        for k, v in d.items():
            self.plot(k, v)

    def img_many(self, d):
        for k, v in d.items():
            self.img(k, v)

    def plot(self, name, y):
        """
        self.plot('loss',1.00)
        """
        x = self.index.get(name, 0)
        self.vis.line(Y=np.array([y]), X=np.array([x]),
                      win=name,
                      opts=dict(title=name),
                      update=None if x == 0 else 'append'
                      )
        self.index[name] = x + 1

    def img(self, name, img_):
        """
        self.img('input_img',t.Tensor(64,64))
        """

        if len(img_.size()) < 3:
            img_ = img_.cpu().unsqueeze(0)
        self.vis.image(img_.cpu(),
                       win=name,
                       opts=dict(title=name)
                       )

    def img_grid_many(self, d):
        for k, v in d.items():
            self.img_grid(k, v)

    def img_grid(self, name, input_3d):
        """
        Turning a batch of images to a grid 
        e.g. input shape: (36, 64, 64)
        Will be a grid of 6x6, each grid is 
        an image size 64x64
        """
        self.img('key', tv.utils.make_grid(
                    input_3d.cpu().unsqueeze(1), pad_value=1, nrow=2))

    def log(self, info, win='log_text'):
        """
        self.log({'loss':1,'lr':0.0001})
        """

        self.log_text += ('[{time}] {info} <br>'.format(
            time=time.strftime('%m.%d_%H:%M:%S'),
            info=info))
        self.vis.text(self.log_text, win=win)

    def __getattr__(self, name):
        return getattr(self.vis, name)

