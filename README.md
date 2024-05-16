# RUL-Prediction

RUL-Prediction is a custom probabilistic framework for forecasing remaining useful life (RUL) by utilizing diffusion models conditioned on transformers. 

RUL prediction can be reduced to solving a regression problem: it is necessary to estimate the number of operating cycles before equipment failure. Existing methods, such as [DAST](https://arxiv.org/abs/2106.15842), use transformer architecture to obtain embeddings that store information about sensor values for previous cycles. To predict the RUL, a fully connected neural network is applied to the resulting embeddings. 

The proposed approach uses a diffusion model to solve the regression problem, where embeddings from the transformer mentioned earlier are used as regressors. The mathematical formulation of such a model and its architecture are taken from [CARD](https://arxiv.org/abs/2206.07275) paper.

# Installation

```
!pip install git+https://github.com/sashaatlasov/RUL_Prediction.git
```

# Quick start 

```python
from rulpred.RULprediction import RULpredictor
from rulpred.training import *

from hparams import config

model = RULpredictor(win_size)
model.to(device)
optim = torch.optim.Adam(model.parameters(), lr=config["learning_rate_start"])

for i in range(1, config["epochs"] + 1):
    loss = train_epoch(model, train_dataloader, optim, device=device)
    if i % 5 == 0:
        val_loss = validate(model, val_dataloader, device)
test_loss = validate(model, test_dataloader, device)
```

# Experimental results

The experiments were performed on datasets from the set. Below are the metrics for each.

|      |  RMSE | nRMSEp |  NLPD | PICP |  Score  |
|:----:|:-----:|:------:|:-----:|:----:|:-------:|
| F001 | 13.22 |  1.18  | -0.87 | 0.88 |  357.21 |
| F002 | 25.42 |  1.58  |  0.02 | 0.68 | 3033.88 |
| F003 | 11.95 |  1.13  | -0.99 | 0.87 |  372.90 |
| F004 | 20.23 |  1.12  | -0.40 | 0.87 | 2424.05 |

Comparison of trajectories for DAST and RULpredictor on F001 dataset (random units).

![png](https://github.com/sashaatlasov/RUL_Prediction/blob/main/example/pictures/F001_unit10.png)

![png](https://github.com/sashaatlasov/RUL_Prediction/blob/main/example/pictures/F001_unit8.png)

![png](https://github.com/sashaatlasov/RUL_Prediction/blob/main/example/pictures/F001_unit24.png)

![png](https://github.com/sashaatlasov/RUL_Prediction/blob/main/example/pictures/F001_unit99.png)

