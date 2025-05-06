import torch
import torch.nn as nn
from .base_method import Base_method
from openstl.models.fnolstm_model import FNOLSTM_B_Model

from torchmetrics import MetricCollection
from torchmetrics.regression import MeanAbsoluteError, MeanSquaredError

class FNOLSTM(Base_method):
    r"""

    Hybrid LSTM with Fourier Neural Operator

    """

    def _get_teacher_forcing_prob(self, current_epoch: int, total_epochs: int, initial_prob: float=1.0, final_prob: float=0.0, initial_epochs: int=5, final_epochs: int=5) -> float:
        if current_epoch < initial_epochs:
            return initial_prob
        if (total_epochs - current_epoch) < final_epochs:
            return final_prob
        # Linear 
        return initial_prob - (initial_prob - final_prob) * (current_epoch / (total_epochs - initial_epochs - final_epochs))

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
    
    def _build_model(self, **kwargs) -> nn.Module:

        return FNOLSTM_B_Model(**kwargs)

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
        batch_in = torch.cat((batch_x, batch_y), axis=2)

        # Teacher forcing
        current_epoch = self.current_epoch
        total_epochs = self.trainer.max_epochs
        teacher_forcing_prob = self._get_teacher_forcing_prob(current_epoch, total_epochs, initial_epochs=10, final_epochs=20)
        out = self.model(batch_in, teacher_forcing_prob=teacher_forcing_prob)
        loss = self.criterion(out, batch_in[:, :, 1:])
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)

        return loss
    
    def validation_step(self, batch, batch_idx):
        batch_x, batch_y = batch
        batch_x = batch_x.permute(0, 2, 1, 3, 4)
        batch_y = batch_y.permute(0, 2, 1, 3, 4)
        len_y = batch_y.shape[2]
        batch_in = torch.cat((batch_x, batch_y), axis=2)
   
        pred_y = self.model(batch_in, teacher_forcing_prob=0.0)
        loss = self.criterion(pred_y[:, :, -len_y:], batch_y)

        metrics = MetricCollection({
            "val_mse": MeanSquaredError(),
            "val_mae": MeanAbsoluteError(),
        })

        metrics_eval =  metrics(pred_y[:, :, -len_y:].cpu().flatten(), batch_y.cpu().flatten())

        self.log_dict(metrics_eval, on_step=True, on_epoch=True, prog_bar=False)
        self.log('val_loss', loss, on_step=True, on_epoch=True, prog_bar=False)

        return loss
    
    def test_step(self, batch, batch_idx):
        batch_x, batch_y = batch
        batch_x = batch_x.permute(0, 2, 1, 3, 4)
        batch_y = batch_y.permute(0, 2, 1, 3, 4)
        len_y = batch_y.shape[2]
        batch_in = torch.cat((batch_x, batch_y), axis=2)
        pred_y = self.model(batch_in, teacher_forcing_prob=0.0)
        outputs = {'inputs': batch_x.permute(0, 2, 1, 3, 4).cpu().numpy(), 'preds': pred_y[:, :, -len_y:].permute(0, 2, 1, 3, 4).cpu().numpy(), 'trues': batch_y.permute(0, 2, 1, 3, 4).cpu().numpy()}
        self.test_outputs.append(outputs)

        return outputs
    