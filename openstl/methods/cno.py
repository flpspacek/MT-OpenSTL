import torch
import torch.nn as nn
from .base_method import Base_method
from openstl.models import CNO3d_Model

from torchmetrics import MetricCollection
from torchmetrics.regression import MeanAbsoluteError, MeanSquaredError


class CNO(Base_method):
    r"""

    Convolutional Neural Operator
    <https://arxiv.org/abs/2302.01178>

    """

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
    
    def _build_model(self, **kwargs) -> nn.Module:

        return CNO3d_Model(**kwargs)

    def forward(self, batch_x, batch_y=None, **kwargs):
        '''
        - Expects tensor of shape: batch_size, channels, temporal, spatial_1, spatial_2
        '''
        out = self.model(batch_x)

        return out

    def training_step(self, batch, batch_idx):
        batch_x, batch_y = batch

        batch_x = batch_x.permute(0, 2, 1, 3, 4)
        batch_y = batch_y.permute(0, 2, 1, 3, 4)

        out = self.model(batch_x)

        #ic(out.shape)
        #ic(batch_y.shape)

        loss = self.criterion(out, batch_y)
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)

        return loss
    
    def validation_step(self, batch, batch_idx):
        batch_x, batch_y = batch
        batch_x = batch_x.permute(0, 2, 1, 3, 4)
        batch_y = batch_y.permute(0, 2, 1, 3, 4)
   
        pred_y = self(batch_x, batch_y)
        loss = self.criterion(pred_y, batch_y)

        metrics = MetricCollection({
            "val_mse": MeanSquaredError(),
            "val_mae": MeanAbsoluteError(),
        })

        metrics_eval =  metrics(pred_y.cpu().flatten(), batch_y.cpu().flatten()) #metrics(pred_y.flatten().to(self.device), batch_y.flatten().to(self.device))

        self.log_dict(metrics_eval, on_step=True, on_epoch=True, prog_bar=False)
        self.log('val_loss', loss, on_step=True, on_epoch=True, prog_bar=False)

    def test_step(self, batch, batch_idx):
        batch_x, batch_y = batch
        batch_x = batch_x.permute(0, 2, 1, 3, 4)
        batch_y = batch_y.permute(0, 2, 1, 3, 4)
        pred_y = self(batch_x, batch_y)
        outputs = {'inputs': batch_x.permute(0, 2, 1, 3, 4).cpu().numpy(), 'preds': pred_y.permute(0, 2, 1, 3, 4).cpu().numpy(), 'trues': batch_y.permute(0, 2, 1, 3, 4).cpu().numpy()}
        self.test_outputs.append(outputs)

        return outputs
