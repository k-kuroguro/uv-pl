import hydra
import lightning as L
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

from utils import (
    RankedLogger,
    get_config_path,
    instantiate_callbacks,
    instantiate_loggers,
    register_custom_resolvers,
)

log = RankedLogger(__name__, rank_zero_only=True)

register_custom_resolvers()


def train(cfg: DictConfig) -> None:
    resolved_cfg = OmegaConf.to_container(cfg, resolve=True)

    log.info(f"Instantiating model: {cfg.model._target_}")
    model: L.LightningModule = instantiate(cfg.model)

    log.info(f"Instantiating datamodule: {cfg.data._target_}")
    datamodule: L.LightningDataModule = instantiate(cfg.data)

    log.info("Instantiating loggers ...")
    loggers = instantiate_loggers(cfg)
    for logger in loggers:
        logger.log_hyperparams(resolved_cfg)  # type: ignore

    log.info("Instantiating callbacks ...")
    callbacks = instantiate_callbacks(cfg)

    log.info(f"Instantiating trainer: {cfg.trainer._target_}")
    trainer = instantiate(cfg.trainer, callbacks=callbacks, logger=loggers)

    trainer.fit(model, datamodule)

    if loggers and "wandb" in cfg.logger.keys():
        import wandb

        if wandb.run:
            wandb.finish()


@hydra.main(version_base="1.3", config_path=str(get_config_path()), config_name="train.yaml")
def main(cfg: DictConfig) -> None:
    train(cfg)


if __name__ == "__main__":
    main()
