import torch


class SeqNet(torch.nn.Module):
    """1D CNN over the 2-channel window [x_i(t), dx_j(t+1)] of one input-output pair, plus self/action flags (module-level so worker processes can pickle it)"""
    def __init__(self):
        super().__init__()
        self.c = torch.nn.Sequential(torch.nn.Conv1d(2, 16, 5, padding=2), torch.nn.ReLU(), torch.nn.Conv1d(16, 16, 5, padding=2), torch.nn.ReLU())
        self.h = torch.nn.Sequential(torch.nn.Linear(32 + 2, 32), torch.nn.Tanh(), torch.nn.Linear(32, 1))

    def forward(self, s, f):
        z = self.c(s)
        return self.h(torch.cat([z.mean(-1), z.amax(-1), f], 1))[:, 0]
