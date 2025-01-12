import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from torch.distributions.categorical import Categorical
from tqdm import tqdm
from matplotlib import pyplot as plt


# From: https://github.com/andrew-cr/discrete_flow_models/blob/main/notebooks/toycode_masking.ipynb
def test_masking():
    
    # training
    B = 32 # batch size
    D = 10 # dimension
    S = 3 # state space

    class Model(nn.Module):
        def __init__(self, D, S):
            super().__init__()
            self.embedding = nn.Embedding(S, 16)
            self.net = nn.Sequential(
                nn.Linear(17 * D, 128),
                nn.ReLU(),
                nn.Linear(128, 128),
                nn.ReLU(),
                nn.Linear(128, (S-1)*D),
            )

        def forward(self, x, t):
            B, D = x.shape
            x_emb = self.embedding(x) # (B, D, 16)
            net_input = torch.cat([x_emb, t[:, None, None].repeat(1, D, 1)], dim=-1).reshape(B, -1) # (B, D * 17)
            return self.net(net_input).reshape(B, D, S-1) # (B, D, S-1)

    model = Model(D, S)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)

    losses = []
    for _ in tqdm(range(100000)):
        num_ones = torch.randint(0, D+1, (B,))
        x1 = (torch.arange(D)[None, :] < num_ones[:, None]).long()
        # x1 e.g. [1, 1, 1, 0, 0, 0, 0, 0, 0, 0] or [1, 1, 1, 1, 1, 1, 1, 1, 1, 0]

        optimizer.zero_grad()
        t = torch.rand((B,))
        xt = x1.clone()
        xt[torch.rand((B,D)) < (1 - t[:, None])] = S-1 # Corrupt with masks, assume 0, 1, ..., S-2 are the valid values and S-1 represents MASK
        
        # The model outputs logits only over the valid values, we know x1 contains no masks!
        logits = model(xt, t) # (B, D, S-1)

        x1[xt != S-1] = -1 # don't compute the loss on dimensions that are already revealed
        loss = F.cross_entropy(logits.transpose(1,2), x1, reduction='mean', ignore_index=-1)

        loss.backward()
        optimizer.step()
        losses.append(loss.item())


    # Sampling
    t = 0.0
    dt = 0.001
    num_samples = 1000
    noise = 10 # noise * dt * D is the average number of dimensions that get re-masked each timestep
    xt = (S-1) * torch.ones((num_samples, D), dtype=torch.long)

    while t < 1.0:
        logits = model(xt, t * torch.ones((num_samples,))) # (B, D, S-1)
        x1_probs = F.softmax(logits, dim=-1) # (B, D, S-1)
        x1 = Categorical(x1_probs).sample() # (B, D)
        will_unmask = torch.rand((num_samples, D)) < (dt * (1 + noise * t) / (1-t)) # (B, D)
        will_unmask = will_unmask * (xt == (S-1)) # (B, D)
        will_mask = torch.rand((num_samples, D)) < dt * noise # (B, D)
        will_mask = will_mask * (xt != (S-1)) # (B, D)
        xt[will_unmask] = x1[will_unmask]

        t += dt

        if t < 1.0:
            xt[will_mask] = S-1

    # print(samples)
    counts = xt.sum(dim=1).float()
    plt.hist(counts.numpy(), bins=range(D+2))
    plt.show()