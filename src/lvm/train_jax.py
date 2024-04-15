import jax
import jax.numpy as jnp
import optax

# from vit_jax_flax.vit import ViT
from jax import Array, random
import flax
from flax.training import train_state, checkpoints

import torch
import torchvision
from torchvision import transforms
from torchvision.datasets import CIFAR10
from torch.utils.tensorboard import SummaryWriter

import numpy as np
from collections import defaultdict
from tqdm.auto import tqdm


if __name__ == '__main__':
    main_rng = jax.random.PRNGKey(42)
    x: Array = jnp.ones(shape=(5, 32, 32, 3))

    # ViT
    from lvm.models.vit.jax.model import ViT

    model = ViT(
        patch_size=4,
        embed_dim=256,
        hidden_dim=512,
        n_heads=8,
        drop_p=0.2,
        num_layers=6,
        mlp_dim=1024,
        num_classes=10
    )
    main_rng, init_rng, drop_rng = random.split(main_rng, 3)
    params = model.init({'params': init_rng, 'dropout': drop_rng}, x, train=True)['params']

    from lvm.trainer.jax.trainer import (
        image_to_numpy,
        init_train_state,
        train_model
    )


    # Dataset preparation
    test_transform = image_to_numpy
    # For training, we add some augmentations. Neworks are too powerful and would overfit.
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomResizedCrop((32, 32), scale=(0.8, 1.0), ratio=(0.9, 1.1)),
        image_to_numpy
    ])

    train_dataset = CIFAR10('data', train=True, transform=train_transform, download=True)
    val_dataset = CIFAR10('data', train=True, transform=test_transform, download=True)
    train_set, _ = torch.utils.data.random_split(train_dataset, [45000, 5000], generator=torch.Generator().manual_seed(42))
    _, val_set = torch.utils.data.random_split(val_dataset, [45000, 5000], generator=torch.Generator().manual_seed(42))
    test_set = CIFAR10('data', train=False, transform=test_transform, download=True)

    from lvm.trainer.jax.trainer import numpy_collate
    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=128, shuffle=True, drop_last=True, collate_fn=numpy_collate, num_workers=8, persistent_workers=True
    )
    val_loader = torch.utils.data.DataLoader(
        val_set, batch_size=128, shuffle=False, drop_last=False, collate_fn=numpy_collate, num_workers=4, persistent_workers=True
    )
    test_loader = torch.utils.data.DataLoader(
        test_set, batch_size=128, shuffle=False, drop_last=False, collate_fn=numpy_collate, num_workers=4, persistent_workers=True
    )

    # Training ViT
    # logger = SummaryWriter(log_dir='vit_jax_logs')
    state = init_train_state(model, params, 3e-4)
    train_model(train_loader, val_loader, model, state, main_rng, num_epochs=10)
