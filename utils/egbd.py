import torch
import torch.nn as nn
import torch.nn.functional as F

class EvidentialLayer(nn.Module):
    """
    Outputs the parameters of a Dirichlet distribution (alpha).
    Alpha > 1 indicates evidence for the state/action utility.
    """
    def __init__(self, input_dim, output_dim):
        super(EvidentialLayer, self).__init__()
        self.dense = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        # We use Softplus to ensure alpha > 1 (alpha = evidence + 1)
        # evidence = softplus(output)
        # alpha = evidence + 1
        return F.softplus(self.dense(x)) + 1.0

class BeliefDiffusion(nn.Module):
    """
    Propagates evidence across the graph using a diffusion-like mechanism
    on the latent Dirichlet parameters.
    """
    def __init__(self, embedding_dim, n_heads=4):
        super(BeliefDiffusion, self).__init__()
        self.attention = nn.MultiheadAttention(embedding_dim, n_heads, batch_first=True)
        self.norm = nn.LayerNorm(embedding_dim)
        
    def forward(self, x, adj_mask=None):
        # x: [batch, n_nodes, embedding_dim]
        # adj_mask: [batch, 1, n_nodes] or [batch, n_nodes, n_nodes]
        
        # Self-attention as a form of belief propagation
        # We treat attention weights as the diffusion coefficients
        if adj_mask is not None:
            if adj_mask.dim() == 3 and adj_mask.size(1) == 1:
                # [batch, 1, n_nodes] -> [batch, n_nodes, n_nodes]
                attn_mask = adj_mask.repeat(1, x.size(1), 1)
            else:
                attn_mask = adj_mask
        else:
            attn_mask = None
            
        residual = x
        x, _ = self.attention(x, x, x, key_padding_mask=adj_mask.squeeze(1) if adj_mask is not None else None)
        x = self.norm(x + residual)
        return x, None

def kl_divergence_dirichlet(alpha, target_alpha):
    """
    KL Divergence between two Dirichlet distributions.
    Used for uncertainty calibration.
    """
    # Simplified KL for Dirichlet
    # KL(Dir(alpha) || Dir(target_alpha))
    sum_alpha = torch.sum(alpha, dim=-1, keepdim=True)
    sum_target = torch.sum(target_alpha, dim=-1, keepdim=True)
    
    ln_gamma_sum_alpha = torch.lgamma(sum_alpha)
    ln_gamma_sum_target = torch.lgamma(sum_target)
    
    sum_ln_gamma_alpha = torch.sum(torch.lgamma(alpha), dim=-1, keepdim=True)
    sum_ln_gamma_target = torch.sum(torch.lgamma(target_alpha), dim=-1, keepdim=True)
    
    digamma_sum_alpha = torch.digamma(sum_alpha)
    digamma_alpha = torch.digamma(alpha)
    
    kl = (ln_gamma_sum_alpha - ln_gamma_sum_target - 
          sum_ln_gamma_alpha + sum_ln_gamma_target + 
          torch.sum((alpha - target_alpha) * (digamma_alpha - digamma_sum_alpha), dim=-1, keepdim=True))
    return kl
