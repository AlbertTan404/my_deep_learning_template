import torch
from .model_base import ModelBase


class ToyModel(ModelBase):
    def __init__(
        self,
        model_kwargs,
        training_kwargs,
        all_config = None,
    ):
        super().__init__(model_kwargs=model_kwargs, training_kwargs=training_kwargs, all_config=all_config)

        self.model = torch.nn.Sequential(
            torch.nn.Linear(128, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 1)
        )
    
    def forward(self, x):
        return self.model(x)
    
    def get_log_dict(self, batch, split, batch_idx=None):
        x = batch['x']
        y = batch['y']
        y_hat = self.forward(x).squeeze()

        loss = torch.nn.functional.mse_loss(y, y_hat)

        log_dict = {
            f'{split}/mse_loss': loss,
            f'{split}/total_loss': loss,
        }
        return log_dict

    def test_step(self, batch, batch_idx=None):
        x = batch['x']
        y = batch['y']
        y_hat = self.forward(x)

        mse = torch.nn.functional.mse_loss(y, y_hat)

        return {
            'mse': mse
        }
