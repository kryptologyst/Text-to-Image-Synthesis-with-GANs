#!/usr/bin/env python3
"""Training script for text-to-image GAN."""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import os
from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger

from src.data.cifar10_captions import CIFAR10CaptionsDataModule
from src.models.lightning_module import TextToImageGANModule
from src.utils.device import get_device, set_seed, print_device_info


@hydra.main(version_base=None, config_path="src/configs", config_name="config")
def main(cfg: DictConfig) -> None:
    """Main training function."""
    # Set seed for reproducibility
    set_seed(cfg.seed)
    
    # Print device information
    print_device_info()
    
    # Create directories
    os.makedirs(cfg.paths.checkpoint_dir, exist_ok=True)
    os.makedirs(cfg.paths.output_dir, exist_ok=True)
    os.makedirs(cfg.paths.log_dir, exist_ok=True)
    
    # Initialize data module
    data_module = CIFAR10CaptionsDataModule(**cfg.data)
    data_module.setup()
    
    # Initialize model
    model = TextToImageGANModule(
        model_config=cfg.model,
        training_config=cfg.training,
        evaluation_config=cfg.evaluation,
        paths_config=cfg.paths,
        logging_config=cfg.logging,
        device=cfg.device
    )
    
    # Setup callbacks
    callbacks = []
    
    # Model checkpoint callback
    checkpoint_callback = ModelCheckpoint(
        dirpath=cfg.paths.checkpoint_dir,
        filename="text-to-image-gan-{epoch:02d}-{val_loss:.2f}",
        monitor="val_loss",
        mode="min",
        save_top_k=3,
        save_last=True,
        every_n_epochs=cfg.logging.save_every_n_epochs
    )
    callbacks.append(checkpoint_callback)
    
    # Learning rate monitor
    lr_monitor = LearningRateMonitor(logging_interval="epoch")
    callbacks.append(lr_monitor)
    
    # Setup logger
    logger = None
    if cfg.logging.use_wandb:
        logger = WandbLogger(
            project=cfg.logging.wandb_project,
            name=f"{cfg.experiment.name}_{cfg.experiment.version}",
            tags=cfg.experiment.tags
        )
    else:
        logger = TensorBoardLogger(
            save_dir=cfg.paths.log_dir,
            name=cfg.experiment.name,
            version=cfg.experiment.version
        )
    
    # Initialize trainer
    trainer = pl.Trainer(
        max_epochs=cfg.training.max_epochs,
        gradient_clip_val=cfg.training.gradient_clip_val,
        accumulate_grad_batches=cfg.training.accumulate_grad_batches,
        val_check_interval=cfg.training.val_check_interval,
        limit_val_batches=cfg.training.limit_val_batches,
        callbacks=callbacks,
        logger=logger,
        devices="auto",
        accelerator="auto",
        precision="16-mixed" if cfg.device != "cpu" else "32",
        deterministic=True,
        benchmark=False
    )
    
    # Train model
    trainer.fit(model, data_module)
    
    # Save final model
    final_model_path = os.path.join(cfg.paths.checkpoint_dir, "final_model.ckpt")
    trainer.save_checkpoint(final_model_path)
    
    print(f"Training completed! Model saved to {final_model_path}")


if __name__ == "__main__":
    main()
