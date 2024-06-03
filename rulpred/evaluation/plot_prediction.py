import torch
import numpy as np
import matplotlib.pyplot as plt


def compare(unit_idx, model, dast, X_test, y_test, masks):
    mask = np.where(masks == unit_idx)[0]
    X = torch.tensor(X_test[mask], dtype=torch.float32).squeeze(-1)
    y = y_test[mask]
    samples = model.sample(X, 100).numpy() * model.rul_max
    y_mean = np.mean(samples, axis=1).reshape(-1, 1)
    y_dast = dast(X).detach().numpy() * model.rul_max
    return y, y_mean, samples, y_dast

def plot_traj(unit_idx, model, dast, X_test, y_test, masks, save=False):
    y, y_mean, samples, y_dast = compare(unit_idx, model, dast, X_test, y_test, masks)
    y5 = np.squeeze(np.quantile(samples, 0.05, axis=1).reshape(-1, 1), -1)
    y95 = np.squeeze(np.quantile(samples, 0.95, axis=1).reshape(-1, 1), -1)
    
    plt.figure(figsize=(10, 6), dpi=100)
    plt.plot(y, label='Original')
    plt.plot(y_mean, label='Custom prediction', color='red')
    plt.plot(y_dast, label='DAST')
    plt.fill_between(x=range(len(y)), y1=y5, y2=np.squeeze(y_mean, -1), alpha=0.3, color='red')
    plt.fill_between(x=range(len(y)), y1=np.squeeze(y_mean, -1), y2=y95, alpha=0.3, color='red')
    plt.xlabel('Time steps')
    plt.ylabel('RUL')
    plt.xlim(left=0)
    plt.grid(axis='y')
    plt.ylim(bottom=0, top=y[0] + 40)
    plt.title(f'F001 Testing Engine unit{unit_idx}')
    plt.legend()
    if save:
        plt.savefig(f'pictures/F001_unit{unit_idx}', bbox_inches='tight', pad_inches=0, dpi=100)
    plt.show();
