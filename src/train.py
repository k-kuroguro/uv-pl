import logging

import hydra
import lightning as L
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

from utils import get_config_path, register_custom_resolvers

log = logging.getLogger(__name__)

register_custom_resolvers()


def train(cfg: DictConfig) -> None:
    log.info(f"Instantiating {cfg.model._target_} as model")
    model = instantiate(cfg.model)
    datamodule = instantiate(cfg.data)
    if cfg.get("logger", None):
        loggers = [instantiate(c) for c in cfg.logger.values()]
        for logger in loggers:
            logger.log_hyperparams(OmegaConf.to_container(cfg, resolve=True))
        logger = loggers
        print(loggers)
    else:
        logger = False
    callbacks = [instantiate(c) for c in cfg.callbacks.values()]
    trainer = instantiate(cfg.trainer, callbacks=callbacks, logger=logger)
    trainer.fit(model, datamodule)
    # trainer.test()
    if logger:
        for logger in loggers:
            if isinstance(logger, L.pytorch.loggers.WandbLogger):
                import wandb

                if wandb.run:
                    wandb.finish()


@hydra.main(version_base="1.3", config_path=str(get_config_path()), config_name="train.yaml")
def main(cfg: DictConfig) -> None:
    log.info("Start training")
    OmegaConf.resolve(cfg)
    print(OmegaConf.to_yaml(cfg))
    train(cfg)


if __name__ == "__main__":
    main()
