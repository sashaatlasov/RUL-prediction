import torch
import torch.nn.functional as F
import numpy as np
from scipy import stats
from tqdm.auto import tqdm


def picp_metric(flux, flux_pred, flux_err_pred, alpha=0.90):
    p_left, p_right = stats.norm.interval(confidence=alpha, loc=flux_pred, scale=flux_err_pred)
    metric = (flux > p_left) * (flux <= p_right)
    return metric.mean()


def nlpd_metric(flux, flux_pred, flux_err_pred):
    
    metric = (flux - flux_pred) ** 2 / (2 * flux_err_pred**2) + np.log(flux_err_pred) + 0.5 * np.log(2 * np.pi)
    
    return metric.mean()


def nrmse_p_metric(flux, flux_pred, flux_err_pred):

    metric = (flux - flux_pred) ** 2 / flux_err_pred**2
    
    return np.sqrt(metric.mean())


def score(y_pred, y):
    diff = y_pred - y
    sum1 = torch.sum(torch.exp(diff[diff >= 0] / 10)) - 1
    sum2 = torch.sum(torch.exp(-diff[diff < 0] / 13)) - 1
    return sum1 + sum2


def validate(model, dataloader, device):
    model.eval()
    with torch.no_grad():
        se = None
        pbar = tqdm(dataloader)
        rm = model.rul_max
        for t, s in pbar:
            preds = model.sample(s, 100)
            mean = preds.mean(dim=1)
            std = preds.std(dim=1)
            loss = F.mse_loss(mean * rm, t * rm, reduction='none')
            rmse = torch.sqrt(torch.mean(loss))
            print('RMSE:', rmse.item())
            
            loss1 = picp_metric(t, mean, std)
            loss2 = nlpd_metric(t, mean, std)
            loss3 = nrmse_p_metric(t, mean, std)
            loss4 = score(mean * rm, t * rm)
            
        print(f'PICP: {loss1}\nNLPD: {loss2}\nnRMSEp: {loss3}\nScore: {loss4}')
        return 