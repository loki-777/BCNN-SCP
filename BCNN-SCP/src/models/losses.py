import torch

def KL_DIV(mu_p, sig_p_inv, sig_p_logdet, mu_q, sig_q):
    num_blocks = sig_q.shape[0]
    mu_dim = sig_q.shape[1]

    sig_q_logdet = torch.logdet(sig_q)  # shape: (num_blocks,)
    A = sig_q_logdet - sig_p_logdet  # shape: (num_blocks,)

    # Broadcasting sig_p_inv across the first dimension of sig_q
    B = torch.einsum('bij,ij->b', sig_q, sig_p_inv)  # shape: (num_blocks,)

    C = mu_q.view(num_blocks, mu_dim) - mu_p  # shape: (num_blocks, mu_dim)
    C_term = torch.einsum('bi,ij,bj->b', C, sig_p_inv, C)  # shape: (num_blocks,)

    # Combining the terms
    loss = A + mu_dim - B - C_term  # shape: (num_blocks,)
    kl = loss.sum()  # scalar

    return -0.5*kl
