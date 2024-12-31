import warnings

import hydra
import lightning as L
import rootutils
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

from utils import (
    RankedLogger,
    find_project_root,
    get_config_path,
    instantiate_callbacks,
    instantiate_loggers,
    register_custom_resolvers,
)

log = RankedLogger(__name__, rank_zero_only=True)

register_custom_resolvers()

rootutils.set_root(
    path=find_project_root(), project_root_env_var=True, dotenv=True, pythonpath=True
)


@hydra.main(version_base="1.3", config_path=str(get_config_path()), config_name="train.yaml")
def main(cfg: DictConfig) -> float | None:
    if cfg.get("ignore_warnings"):
        log.info("Ignoring warnings ...")
        warnings.filterwarnings("ignore")

    if cfg.get("seed"):
        log.info(f"Setting seed: {cfg.seed}")
        L.seed_everything(cfg.seed, workers=True)

    log.info(f"Instantiating model: {cfg.model._target_}")
    model: L.LightningModule = instantiate(cfg.model)

    log.info(f"Instantiating datamodule: {cfg.data._target_}")
    datamodule: L.LightningDataModule = instantiate(cfg.data)

    log.info("Instantiating loggers ...")
    loggers = instantiate_loggers(cfg)
    resolved_cfg = OmegaConf.to_container(cfg, resolve=True)
    for logger in loggers:
        logger.log_hyperparams(resolved_cfg)  # type: ignore

    log.info("Instantiating callbacks ...")
    callbacks = instantiate_callbacks(cfg)

    log.info(f"Instantiating trainer: {cfg.trainer._target_}")
    trainer = instantiate(cfg.trainer, callbacks=callbacks, logger=loggers)

    if cfg.get("train"):
        if cfg.get("ckpt_path"):
            log.info(f"Resuming training from checkpoint: {cfg.ckpt_path}")
        else:
            log.info("Starting training ...")
        trainer.fit(model=model, datamodule=datamodule, ckpt_path=cfg.get("ckpt_path"))
        log.info("Training is done.")

    train_metrics = trainer.callback_metrics

    if cfg.get("test"):
        log.info("Starting testing ...")
        ckpt_path = trainer.checkpoint_callback.best_model_path
        if ckpt_path == "":
            log.warning("Best ckpt not found. Using current weights for testing ...")
            ckpt_path = None
        else:
            log.info(f"Using best checkpoint: {ckpt_path}")
        trainer.test(model=model, datamodule=datamodule, ckpt_path=ckpt_path)
        log.info("Testing is done.")

    test_metrics = trainer.callback_metrics

    if cfg.logger and "wandb" in cfg.logger.keys():
        import wandb

        if wandb.run:
            wandb.finish()

    overall_metrics = {**train_metrics, **test_metrics}
    target_metric = cfg.get("target_metric")
    if not target_metric:
        log.info("No target metric provided.")
        return None
    if target_metric not in train_metrics:
        log.warning(f"Target metric {target_metric} not found in metrics.")
        return None
    log.info(f"Returning {target_metric} for optimization.")
    return overall_metrics[target_metric]


if __name__ == "__main__":
    main()
