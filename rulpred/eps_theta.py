import torch
import torch.nn as nn
import torch.nn.functional as F


class ConditionalLinear(nn.Module):
    def __init__(self, num_in, num_out, n_steps):
        super(ConditionalLinear, self).__init__()
        self.num_out = num_out
        self.lin = nn.Linear(num_in, num_out)
        self.embed = nn.Embedding(n_steps, num_out)
        self.embed.weight.data.uniform_()

    def forward(self, x, t):
        out = self.lin(x)
        gamma = self.embed(t)
        out = gamma.view(-1, self.num_out) * out
        return out
    

class ConditionalGuidedModel(nn.Module):
    def __init__(self, num_timesteps, data_dim, hidden_dim=128):
        super(ConditionalGuidedModel, self).__init__()
        n_steps = num_timesteps + 1
        data_dim = data_dim + 1
        self.lin1 = ConditionalLinear(data_dim, hidden_dim, n_steps)
        self.lin2 = ConditionalLinear(hidden_dim, hidden_dim, n_steps)
        self.lin3 = ConditionalLinear(hidden_dim, hidden_dim, n_steps)
        self.lin4 = nn.Linear(hidden_dim, 1)

    def forward(self, x, y_t, t):
        eps_pred = torch.cat((y_t, x), dim=1)
        eps_pred = F.softplus(self.lin1(eps_pred, t))
        eps_pred = F.softplus(self.lin2(eps_pred, t))
        eps_pred = F.softplus(self.lin3(eps_pred, t))
        return self.lin4(eps_pred)
