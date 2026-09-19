import torch 
import torch.nn as nn
import torch.nn.functional as F

class token_merging(nn.Module):


    def __init__(self,ori_shape,clusters):
        super(token_merging, self).__init__()

        self.shape = ori_shape
        self.alpha = nn.Parameter(torch.tensor(1.0))
        self.beta  = nn.Parameter(torch.tensor(1.0))
        self.gamma = nn.Parameter(torch.tensor(1.0))
        self.Smax = ori_shape[1]
        self.Tmax = ori_shape[3]
        K=clusters
        dim = 2048
        self.centers = torch.nn.Parameter(torch.randn(ori_shape[0], K, ori_shape[-1]))
        # False reproduces the released "nosim" checkpoints (feature-cosine clustering only).
        # Set True to enable the composite spatio-temporal similarity of Eq. 1 (full LTM).
        self.use_composite_sim = False
        self.center_coord = torch.nn.Parameter(torch.randn(ori_shape[0],K,3))

    def ordering(self,tokens):

        y, x, t = torch.meshgrid(torch.arange(self.shape[1]), torch.arange(self.shape[2]), torch.arange(self.shape[3]), indexing="ij")

        x_flat = x.flatten()  
        y_flat = y.flatten()  
        t_flat = t.flatten()  
        labels = torch.stack([x_flat, y_flat, t_flat], dim=1)  #(98, 3)
        return labels
    
    def distance(self,tokens,labels):

        tokens_norm = tokens / (tokens.norm(dim=-1, keepdim=True) + 1e-8)        # (N, dim)
        centers_norm = self.centers / (self.centers.norm(dim=-1, keepdim=True) + 1e-8)  # (K, dim)

        cos_sim = F.cosine_similarity(tokens_norm.unsqueeze(2),centers_norm.unsqueeze(1),dim=-1)
        d_feature = 1.0 - cos_sim

        coords_2d = labels[:, :2].float().cuda()
        coords_2d = coords_2d.unsqueeze(0).expand(self.shape[0], -1, -1)  #(98, 2)
        t_values  = labels[:, 2].cuda()  #(98,)
        t_values = t_values.unsqueeze(0).expand(self.shape[0], -1) 

        coords_2d_c = self.center_coord[:,:,:2].float()
        t_values_c = self.center_coord[:,:,2]

        delta_xy  = torch.cdist(coords_2d, coords_2d_c, p=2)
        #d_spatial = delta_xy.norm(dim=-1)  # (B, N, K)

        d_spatial = delta_xy / self.Smax # normalized

        delta_t   = torch.abs(t_values[:,:, None] - t_values_c[:,None, :])
        d_temporal = delta_t.abs() / self.Tmax  # (B, N, K) normalized


        d_composite = (self.alpha * d_feature
                       + self.beta * d_spatial
                       + self.gamma * d_temporal)
        
        prob = F.softmax(-d_composite, dim=-1)  #(N,K)

        cluster_centers = (prob.unsqueeze(-1) * tokens.unsqueeze(2)).sum(dim=1) / (prob.permute((0,2,1)).sum(dim=-1, keepdim=True) + 1e-8)

        return cluster_centers

        #return d_composite


       

    def cluster(self, tokens):

        tokens_norm = tokens / (tokens.norm(dim=-1, keepdim=True) + 1e-8)        # (N, dim)
        centers_norm = self.centers / (self.centers.norm(dim=-1, keepdim=True) + 1e-8)  # (K, dim)
        cos_sim = F.cosine_similarity(tokens_norm.unsqueeze(2),centers_norm.unsqueeze(1),dim=-1)
        dist = 1.0 - cos_sim

        prob = F.softmax(-dist, dim=-1)  #(N,K)

        cluster_centers = (prob.unsqueeze(-1) * tokens.unsqueeze(2)).sum(dim=1) / (prob.permute((0,2,1)).sum(dim=-1, keepdim=True) + 1e-8)

        return cluster_centers
        
    def forward(self,x):
        if getattr(self, "use_composite_sim", False):
            # full LTM: composite distance over feature + spatial + temporal terms
            labels = self.ordering(x)   # (x,y,t) index per token
            x = self.distance(x, labels)
        else:
            # "nosim": feature-cosine clustering only
            x = self.cluster(x)

        return x 
