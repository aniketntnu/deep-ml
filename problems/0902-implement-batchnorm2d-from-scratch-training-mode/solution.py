import torch

def batchnorm2d(x, gamma, beta, eps=1e-5):
    # TODO: training-mode batchnorm2d


    n,c,h,w = x.shape

    mu = x.mean(dim=(0,2,3),keepdim=True)
    sigma = x.var(dim=(0,2,3),keepdim=True,unbiased=False)

    norm = (x-mu)/torch.sqrt((sigma+eps))

    gamma= gamma.view((1,-1,1,1))
    beta= beta.view((1,-1,1,1))

    y= gamma*norm+beta

    return y




