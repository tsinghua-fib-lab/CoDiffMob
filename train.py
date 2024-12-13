from argparse import ArgumentParser

import pytorch_lightning as pl
import setproctitle
import yaml
from dataset.module import DataModule
from model import RegionDiff
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.strategies import DDPStrategy
from utils.ema import EMA


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--save", type=str, required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)
    pl.seed_everything(cfg["seed"], workers=True)
    setproctitle.setproctitle(cfg["title"])

    model = RegionDiff(cfg)
    datamodule = DataModule(cfg)
    lr_monitor = LearningRateMonitor(logging_interval="epoch")
    tb_logger = pl_loggers.TensorBoardLogger(
        save_dir=args.save, name="logs", version=cfg["version"]
    )
    callbacks = [lr_monitor]
    if cfg["model"]["enable_ema"]:
        ema_callback = EMA(decay=cfg["model"]["ema_decay"])
        callbacks.append(ema_callback)
    trainer = pl.Trainer(
        logger=tb_logger,
        accelerator=cfg["trainer"]["accelerator"],
        devices=cfg["trainer"]["devices"],
        strategy=DDPStrategy(find_unused_parameters=True, gradient_as_bucket_view=True),
        callbacks=callbacks,
        max_epochs=cfg["trainer"]["max_epochs"],
        check_val_every_n_epoch=cfg["trainer"]["val_interval"],
    )
    trainer.fit(model, datamodule, ckpt_path=cfg["trainer"]["ckpt_path"])


if __name__ == "__main__":
    main()
