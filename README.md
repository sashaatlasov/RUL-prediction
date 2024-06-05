# RUL-Prediction

RUL-Prediction is a custom probabilistic framework for forecasing remaining useful life (RUL) by utilizing diffusion models conditioned on transformers. 

RUL prediction can be reduced to solving a regression problem: it is necessary to estimate the number of operating cycles before equipment failure. Existing methods, such as [DAST](https://arxiv.org/abs/2106.15842), use transformer architecture to obtain embeddings that store information about sensor values for previous cycles. To predict the RUL, a fully connected neural network is applied to the resulting embeddings. 

The proposed approach uses a diffusion model to solve the regression problem, where embeddings from the transformer mentioned earlier are used as regressors. The mathematical formulation of such a model and its architecture are taken from [CARD](https://arxiv.org/abs/2206.07275) paper.

# Installation

```
!pip install git+https://github.com/sashaatlasov/RUL-Prediction.git
```

# Quick start 

To work with this library, the input data must have the following dimensions: sensors - `(batch_size, time_steps, input_size)`, target - `(batch_size, target_dim)` An example of creating a dataset and initial work with the library is presented in the example section in the file `example.ipynb`.

Below is the code responsible for initializing and training the model with hyperparameters from `hparams.py`.

```python
from rulpred.RULprediction import RULpredictor
from rulpred.training import *

from hparams import config

def main(k, device='cpu'):
    if k % 2 == 0:
        win_size = 60
    else:
        win_size = 40
        
    model = RULpredictor(win_size, config['betas'], config['num_timesteps'], config['rul_max'],
                        **config['dast_conf'])
    model.to(device)

    optim = torch.optim.Adam(model.parameters(), lr=config["learning_rate_start"])
    train_dataloader, val_dataloader, test_dataloader = get_data(k)

    for i in range(1, config['epochs'] + 1):
        loss = train_epoch(model, train_dataloader, optim, device=device)
        if i % 10 == 0:
            val_loss = validate(model, val_dataloader, device)
    test_loss = validate(model, test_dataloader, device)
    torch.save(model, f'Model_F00{k}.pt')
```

# Experimental results

The experiments were performed on datasets from the [CMAPSS](https://data.nasa.gov/Aerospace/CMAPSS-Jet-Engine-Simulated-Data/ff5v-kuh6/about_data). Below are the metrics for each sub-dataset.

|      |  RMSE | nRMSEp |  NLPD | PICP |  Score  |
|:----:|:-----:|:------:|:-----:|:----:|:-------:|
| F001 | 13.22 |  1.18  | -0.87 | 0.88 |  357.21 |
| F002 | 25.42 |  1.58  |  0.02 | 0.68 | 3033.88 |
| F003 | 11.95 |  1.13  | -0.99 | 0.87 |  372.90 |
| F004 | 20.23 |  1.12  | -0.40 | 0.87 | 2424.05 |

The following code allows to plot trajectories for pre-trained RUL-predictor and DAST:

```python
from rulpred.evaluation.plot_prediction import plot_traj

plot_traj(unit_idx=10, model, dast, X_test, y_test, masks)
```


Comparison of trajectories for DAST and RULpredictor on F001 dataset (random units).

![png](https://github.com/sashaatlasov/RUL-prediction/blob/main/example/pictures/F001_unit10.png)

![png](https://github.com/sashaatlasov/RUL-prediction/blob/main/example/pictures/F001_unit8.png)

![png](https://github.com/sashaatlasov/RUL-prediction/blob/main/example/pictures/F001_unit24.png)

![png](https://github.com/sashaatlasov/RUL-prediction/blob/main/example/pictures/F001_unit99.png)

