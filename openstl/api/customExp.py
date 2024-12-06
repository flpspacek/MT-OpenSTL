from lightning import seed_everything, Trainer
from lightning.pytorch.loggers import WandbLogger

from openstl.api.exp import BaseExperiment

class CustomExperiment(BaseExperiment):
    def __init__(self, args, dataloaders=None, strategy='auto'):
        super().__init__(args, dataloaders, strategy)
    
    def _init_trainer(self, args, callbacks, strategy):
        return Trainer(devices=args.gpus,  # Use these GPUs
                max_epochs=args.epoch,  # Maximum number of epochs to train for
                strategy=strategy,   # 'ddp', 'deepspeed_stage_2', 'ddp_find_unused_parameters_false'
                accelerator='gpu',  # Use distributed data parallel
                callbacks=callbacks,
                precision=args.precision,
                logger=self._init_logger(args)
            )
    
    def _init_logger(self, args):
        return WandbLogger(name=args.logger_name)