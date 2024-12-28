import logging

import hydra
import lightning as L
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

from utils import get_config_path, register_custom_resolvers

logger = logging.getLogger(__name__)

register_custom_resolvers()


def train(cfg: DictConfig) -> None:
    model = instantiate(cfg.model)
    datamodule = instantiate(cfg.data)
    if cfg.get("logger", None):
        logger = instantiate(cfg.logger)
        logger.log_hyperparams(OmegaConf.to_container(cfg, resolve=True))
    else:
        logger = False
    trainer = instantiate(cfg.trainer, callbacks=None, logger=logger)
    trainer.fit(model, datamodule)
    # trainer.test()


@hydra.main(version_base="1.3", config_path=str(get_config_path()), config_name="train.yaml")
def main(cfg: DictConfig) -> None:
    logger.info("Start training")
    OmegaConf.resolve(cfg)
    print(OmegaConf.to_yaml(cfg))
    train(cfg)


if __name__ == "__main__":
    main()
