import torch

class BetaScheduler:
    def __init__(self, beta_init, beta_target, schedules, device=None, dtype=torch.float32):
        assert len(beta_init) == len(beta_target) == len(schedules)
        self.beta_init = torch.tensor(beta_init, dtype=dtype, device=device)
        self.beta_target = torch.tensor(beta_target, dtype=dtype, device=device)
        self.schedules = schedules
        self.device = device
        self.dtype = dtype

    def __call__(self, epoch):
        # Calculate the progress of each schedule (a value from 0.0 to 1.0)
        progress = torch.tensor([s(epoch) for s in self.schedules], dtype=self.dtype, device=self.device)
        
        # Calculate the final beta values using this progress
        beta = self.beta_init + progress * (self.beta_target - self.beta_init)
        
        # Return BOTH the beta values and the progress tensor
        return beta, progress