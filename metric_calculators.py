import torch
from abc import ABC, abstractmethod
import pdb

class MetricCalculator(ABC):
    
    @abstractmethod
    def update(self, batch_size, dx, *feats, **aux_params): return NotImplemented
    
    @abstractmethod
    def finalize(self): return NotImplemented

def compute_correlation(covariance, eps=1e-7):
    std = torch.diagonal(covariance).sqrt()
    covariance = covariance / (torch.clamp(torch.outer(std, std), min=eps))
    return covariance

class CovarianceMetric(MetricCalculator):
    name = 'covariance'
    
    def __init__(self):
        self.std = None
        self.mean = None
        self.outer = None
        self.sos = None
        self.num_updates = 0
        self.bsz = None
    
    def update(self, batch_size, *feats, **aux_params):
        feats = torch.cat(feats, dim=0) #[dim, num_elts]
        feats = torch.nan_to_num(feats, 0, 0, 0)

        mean = feats.sum(dim=1)
        sos = (feats**2).sum(dim=1) # sum of squares
        outer = (feats @ feats.T) 
        
        if self.bsz == None:
            self.bsz = batch_size
        
        if self.mean  is None: self.mean  = torch.zeros_like( mean, dtype=torch.float64)
        if self.outer is None: self.outer = torch.zeros_like(outer, dtype=torch.float64)
        if self.sos is None: self.sos = torch.zeros_like(sos, dtype=torch.float64)
            
        self.mean  += mean  
        self.outer += outer 
        self.sos += sos 
    
        # debugging
        self.num_updates +=1

    def finalize(self, numel, eps=1e-4, dot_prod=False, pca=False, scale_cov=False, normalize=False, print_featnorms=False):
        self.outer = self.outer.div(numel)
        self.mean  = self.mean.div(numel)
        self.sos = torch.sqrt(self.sos)
        #scaling_factor = 1.0 / self.bsz
        
        if dot_prod:
            # this is equivalent to E_ab from git rebasin
            cov = self.outer #* scaling_factor
        else:
            cov = self.outer - torch.outer(self.mean, self.mean) 
        if scale_cov:
            cov *= 1.0 / self.bsz
        if pca:
            new_val = int(0.95 * cov.shape[1])
            U,S,V = torch.pca_lowrank(cov, q=new_val)
            cov = U[:,:new_val] @ torch.diag(S[:new_val]) @ V.T
        if normalize:
            cov = cov / (torch.outer(self.sos, self.sos) + eps)

        if print_featnorms:
            len_feats = len(self.sos) // 2
            mean1 = torch.mean(self.sos[:len_feats]).item()
            std1 = torch.std(self.sos[:len_feats]).item()
            mean2 = torch.mean(self.sos[len_feats:]).item()
            std2 = torch.std(self.sos[len_feats:]).item()
            print(mean1, std1, mean2, std2)
        if torch.isnan(cov).any():
            breakpoint()
        if (torch.diagonal(cov) < 0).sum():
            pdb.set_trace()
        return cov

class MeanMetric(MetricCalculator):
    name = 'mean'
    
    def __init__(self):
        self.mean = None
    
    def update(self, batch_size, *feats, **aux_params):
        feats = torch.cat(feats, dim=0)
        mean = feats.abs().mean(dim=1)
        if self.mean is None: 
            self.mean = torch.zeros_like(mean)
        self.mean  += mean  * batch_size
    
    def finalize(self, numel, eps=1e-4, print_featnorms=False):
        return self.mean / numel
        

def get_metric_fns(names):
    metrics = {}
    for name in names:
        if name == 'mean':
            metrics[name] = MeanMetric
        elif name == 'covariance':
            metrics[name] = CovarianceMetric
        else:
            raise NotImplementedError(name)
    return metrics