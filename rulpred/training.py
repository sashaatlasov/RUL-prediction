import torch
from tqdm import tqdm
import torch.nn.functional as F


def train_step(model, t, s, optimizer, device):
    optimizer.zero_grad()
    t, s = t.to(device), s.to(device)
    loss = model(t, s)
    loss.backward()
    optimizer.step()
    return loss


def train_epoch(model, train_dataloader, optimizer, device, scheduler=None):
    model.train()
    pbar = tqdm(train_dataloader)
    loss_ema = None
    for t, s in pbar:
        train_loss = train_step(model, t, s, optimizer, device)
        loss_ema = train_loss if loss_ema is None else 0.9 * loss_ema + 0.1 * train_loss
        pbar.set_description(f"loss: {loss_ema:.4f}")
    return loss_ema


def validate(model, val_dataloader, device):
    model.eval()
    with torch.no_grad():
        se = None
        pbar = tqdm(val_dataloader)
        for t, s in pbar:
            preds = model.sample(s)
            mean = preds.mean(dim=1)
            loss = F.mse_loss(mean.squeeze() * model.rul_max, t * model.rul_max, reduction='none')
            se = loss if se is None else torch.cat((se, loss), dim=0)
        mse = torch.mean(se)
        return torch.sqrt(mse)
    

