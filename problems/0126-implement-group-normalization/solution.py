import torch

def group_normalization(X: torch.Tensor, gamma: torch.Tensor, beta: torch.Tensor, num_groups: int, epsilon: float = 1e-5) -> torch.Tensor:
    """
    Perform Group Normalization on a 4D input tensor.
    
    Args:
        X: torch tensor of shape (B, C, H, W), input data
        gamma: torch tensor of shape (1, C, 1, 1), scale parameter
        beta: torch tensor of shape (1, C, 1, 1), shift parameter
        num_groups: number of groups for normalization
        epsilon: small constant to avoid division by zero
    
    Returns:
        norm_X: torch tensor of shape (B, C, H, W), normalized output
    """
    x= X
    n,c,h,w = x.shape

    x= x.view(n,num_groups,c//num_groups,h,w)

    mu = x.mean(dim=(2,3,4),keepdim=True)
    sigma = x.var(dim=(2,3,4),keepdim=True,unbiased=False)

    x= (x-mu)/ torch.sqrt(sigma+epsilon)

    x= x.view(n,c,h,w)

    return gamma*x+beta

