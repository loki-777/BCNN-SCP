import torch

def KL_DIV(mu_p, sig_p, mu_q, sig_q):
    kl = 0
    mu_dim = sig_q.shape[1]
    num_blocks = sig_q.shape[0]
    n = sig_q.shape[1]
    mu_p = mu_p.to(sig_q.device)
    sig_p_inv = torch.linalg.pinv(sig_p).to(sig_q.device)
    sig_p_logdet = torch.logdet(sig_p).to(sig_q.device)
    for i in range(num_blocks):
        sig_q_logdet = torch.logdet(sig_q[i,:,:])
        A =  sig_q_logdet - sig_p_logdet
        B = torch.trace(sig_p_inv @ sig_q[i,:,:])
        C = (mu_q[mu_dim*i:mu_dim*(i+1)] - mu_p[mu_dim*i:mu_dim*(i+1)])
        loss = (A + n - B - (C) @ sig_p_inv @ (C))
        kl += loss

    return -0.5*kl
