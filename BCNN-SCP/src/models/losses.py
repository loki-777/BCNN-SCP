import torch


def KL_DIV(mu_p, sig_p_inv, sig_p_logdet, mu_q, sig_q):
    """
    Compute the KL divergence between two block Gaussians.
    Parameters:
        mu_p (torch.Tensor): Mean of the prior, shape: ()
        sig_p_inv (torch.Tensor): Inverse of the covariance matrix of the prior, shape: (w_dim, w_dim)
        sig_p_logdet (torch.Tensor): Log determinant of the covariance matrix of the prior, shape: ()
        mu_q (torch.Tensor): Mean of the posterior, shape: (num_blocks, w_dim)
        sig_q (torch.Tensor): Covariance matrix of the posterior, shape: (num_blocks, w_dim, w_dim)
    """
    w_dim = mu_q.shape[1]

    # log(|sig_q|/|sig_p|)
    sig_q_logdet = torch.logdet(sig_q)  # shape: (num_blocks,)
    A = sig_q_logdet - sig_p_logdet  # shape: (num_blocks,)

    # Tr(sig_p^-1 * sig_q)
    B = torch.einsum("ij,bji->b", sig_p_inv, sig_q)  # shape: (num_blocks,)

    # (mu_q - mu_p)^T * sig_p^-1 * (mu_q - mu_p)
    mu_diff = mu_q - mu_p  # shape: (num_blocks, w_dim)
    C = torch.einsum("bi,ij,bj->b", mu_diff, sig_p_inv, mu_diff)  # shape: (num_blocks,)

    return -0.5 * torch.sum(A + w_dim - B - C)
