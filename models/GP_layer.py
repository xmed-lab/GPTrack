import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torch.autograd import Variable
from torch import Tensor
import cv2
import math
import numpy as np
import time
from numpy.linalg import inv


class GPlayer(nn.Module):


    def __init__(self):
        super(GPlayer, self).__init__()

        self.gamma2 = nn.Parameter(torch.randn(1), requires_grad=True).float()
        self.ell = nn.Parameter(torch.randn(1), requires_grad=True).float()
        self.sigma2 = nn.Parameter(torch.randn(1), requires_grad=True).float()


    def forward(self, D, Y):
        '''
        :param D: Distance matrix
        :param Y: Stacked outputs from encoder
        :return: Z: transformed latent space
        '''
        b, l, p, c = Y.size()
        Y = Y.contiguous()
        Y = Y.view(b, l, -1)

        K = torch.exp(self.gamma2) * (1 + math.sqrt(3) * D / torch.exp(self.ell)) * torch.exp(-math.sqrt(3) * D / torch.exp(self.ell))
        I = torch.eye(l).expand(b, l, l).float().to(Y.device)
        X = torch.linalg.solve(K+torch.exp(self.sigma2)*I, Y)

        Z = K.bmm(X)

        Z = F.relu(Z)

        Z = Z.view(b, l, p, c)

        return Z