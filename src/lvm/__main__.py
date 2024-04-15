# %%
"""
# Training a `ViT` with Jax + `flax`
"""

# %%
import jax
import jax.numpy as jnp
import jax.dlpack
from jax import grad, jit, vmap, random
from jax import random
from jax.example_libraries import stax, optimizers

from tensorflow import keras
import tensorflow_datasets as tfds
import tensorflow as tf

import time
import numpy.random as npr
import math

from typing import Optional

import optax
from flax.training import train_state, checkpoints

from tqdm.auto import tqdm
from collections import defaultdict

import numpy as np

import torch
from torch.utils.tensorboard import SummaryWriter

import matplotlib.pyplot as plt

# %%
"""
## ViT
"""

# %%
from rich import print
from flax import linen as nn

# %%
class Patches(nn.Module):
  patch_size: int
  embed_dim: int

  def setup(self):
    self.conv = nn.Conv(
        features=self.embed_dim,
        kernel_size=(self.patch_size, self.patch_size),
        strides=(self.patch_size, self.patch_size),
        padding='VALID'
    )

  def __call__(self, images):
    patches = self.conv(images)
    b, h, w, c = patches.shape
    patches = jnp.reshape(patches, (b, h*w, c))
    return patches

# %%
class PatchEncoder(nn.Module):
  hidden_dim: int

  @nn.compact
  def __call__(self, x):
    assert x.ndim == 3
    n, seq_len, _ = x.shape
    # Hidden dim
    x = nn.Dense(self.hidden_dim)(x)
    # Add cls token
    cls = self.param('cls_token', nn.initializers.zeros, (1, 1, self.hidden_dim))
    cls = jnp.tile(cls, (n, 1, 1))
    x = jnp.concatenate([cls, x], axis=1)
    # Add position embedding
    pos_embed = self.param(
        'position_embedding', 
        nn.initializers.normal(stddev=0.02), # From BERT
        (1, seq_len + 1, self.hidden_dim)
    )
    return x + pos_embed

# %%
class MultiHeadSelfAttention(nn.Module):
  hidden_dim: int
  n_heads: int
  drop_p: float

  def setup(self):
    self.q_net = nn.Dense(self.hidden_dim)
    self.k_net = nn.Dense(self.hidden_dim)
    self.v_net = nn.Dense(self.hidden_dim)

    self.proj_net = nn.Dense(self.hidden_dim)

    self.att_drop = nn.Dropout(self.drop_p)
    self.proj_drop = nn.Dropout(self.drop_p)

  def __call__(self, x, train=True):
    B, T, C = x.shape # batch_size, seq_length, hidden_dim
    N, D = self.n_heads, C // self.n_heads # num_heads, head_dim
    q = self.q_net(x).reshape(B, T, N, D).transpose(0, 2, 1, 3) # (B, N, T, D)
    k = self.k_net(x).reshape(B, T, N, D).transpose(0, 2, 1, 3)
    v = self.v_net(x).reshape(B, T, N, D).transpose(0, 2, 1, 3)

    # weights (B, N, T, T)
    weights = jnp.matmul(q, jnp.swapaxes(k, -2, -1)) / math.sqrt(D)
    normalized_weights = nn.softmax(weights, axis=-1)

    # attention (B, N, T, D)
    attention = jnp.matmul(normalized_weights, v)
    attention = self.att_drop(attention, deterministic=not train)

    # gather heads
    attention = attention.transpose(0, 2, 1, 3).reshape(B, T, N*D)

    # project
    out = self.proj_drop(self.proj_net(attention), deterministic=not train)

    return out

# %%
class MLP(nn.Module):
  mlp_dim: int
  drop_p: float
  out_dim: Optional[int] = None

  @nn.compact
  def __call__(self, inputs, train=True):
    actual_out_dim = inputs.shape[-1] if self.out_dim is None else self.out_dim
    x = nn.Dense(features=self.mlp_dim)(inputs)
    x = nn.gelu(x)
    x = nn.Dropout(rate=self.drop_p, deterministic=not train)(x)
    x = nn.Dense(features=actual_out_dim)(x)
    x = nn.Dropout(rate=self.drop_p, deterministic=not train)(x)
    return x

# %%
class TransformerEncoder(nn.Module):
  embed_dim: int
  hidden_dim: int
  n_heads: int
  drop_p: float
  mlp_dim: int

  def setup(self):
    self.mha = MultiHeadSelfAttention(self.hidden_dim, self.n_heads, self.drop_p)
    self.mlp = MLP(self.mlp_dim, self.drop_p)
    self.layer_norm = nn.LayerNorm(epsilon=1e-6)
  
  def __call__(self, inputs, train=True):
    # Attention Block
    x = self.layer_norm(inputs)
    x = self.mha(x, train)
    x = inputs + x
    # MLP block
    y = self.layer_norm(x)
    y = self.mlp(y, train)

    return x + y

# %%
class ViT(nn.Module):
  patch_size: int
  embed_dim: int
  hidden_dim: int
  n_heads: int
  drop_p: float
  num_layers: int
  mlp_dim: int
  num_classes: int

  def setup(self):
    self.patch_extracter = Patches(self.patch_size, self.embed_dim)
    self.patch_encoder = PatchEncoder(self.hidden_dim)
    self.dropout = nn.Dropout(self.drop_p)
    self.transformer_blocks = [
      TransformerEncoder(self.embed_dim, self.hidden_dim, self.n_heads, self.drop_p, self.mlp_dim)
      for _ in range(self.num_layers)]
    self.cls_head = nn.Dense(features=self.num_classes)

  def __call__(self, x, train=True):
    x = self.patch_extracter(x)
    x = self.patch_encoder(x)
    x = self.dropout(x, deterministic=not train)
    for block in self.transformer_blocks:
      x = block(x, train)
    # MLP head
    x = x[:, 0] # [CLS] token
    x = self.cls_head(x)
    return x

# %%
"""
## Hyper-parameters
"""

# %%
IMAGE_SIZE = 32
BATCH_SIZE = 128
DATA_MEANS = np.array([0.49139968, 0.48215841, 0.44653091])
DATA_STD = np.array([0.24703223, 0.24348513, 0.26158784])
CROP_SCALES = (0.8, 1.0)
CROP_RATIO = (0.9, 1.1)
SEED = 42

# %%
"""
## Dataset preparation(torchvision)
"""

# %%
import torchvision
from torchvision import transforms
from torchvision.datasets import CIFAR10

# %%
def image_to_numpy(img):
  img = np.array(img, dtype=np.float32)
  img = (img / 255. - DATA_MEANS) / DATA_STD
  return img

# %%
# We need to stack the batch elements
def numpy_collate(batch):
  if isinstance(batch[0], np.ndarray):
    return np.stack(batch)
  elif isinstance(batch[0], (tuple, list)):
    transposed = zip(*batch)
    return [numpy_collate(samples) for samples in transposed]
  else:
    return np.array(batch)

# %%
test_transform = image_to_numpy
# For training, we add some augmentations. Neworks are too powerful and would overfit.
train_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomResizedCrop((IMAGE_SIZE, IMAGE_SIZE), scale=CROP_SCALES, ratio=CROP_RATIO),
    image_to_numpy
])

# Validation set should not use the augmentation.
train_dataset = CIFAR10('data', train=True, transform=train_transform, download=True)
val_dataset = CIFAR10('data', train=True, transform=test_transform, download=True)
train_set, _ = torch.utils.data.random_split(train_dataset, [45000, 5000], generator=torch.Generator().manual_seed(SEED))
_, val_set = torch.utils.data.random_split(val_dataset, [45000, 5000], generator=torch.Generator().manual_seed(SEED))
test_set = CIFAR10('data', train=False, transform=test_transform, download=True)

train_loader = torch.utils.data.DataLoader(
    train_set, batch_size=BATCH_SIZE, shuffle=True, drop_last=True, num_workers=2, persistent_workers=True, collate_fn=numpy_collate,
)
val_loader = torch.utils.data.DataLoader(
    val_set, batch_size=BATCH_SIZE, shuffle=False, drop_last=False, num_workers=2, persistent_workers=True, collate_fn=numpy_collate,
)
test_loader = torch.utils.data.DataLoader(
    test_set, batch_size=BATCH_SIZE, shuffle=False, drop_last=False, num_workers=2, persistent_workers=True, collate_fn=numpy_collate,
)

# %%
batch = next(iter(train_loader))
print(f'image: {batch[0].shape}, label: {batch[1].shape}')
print(f'type batch[0]: {type(batch[0])}, batch[1]: {type(batch[1])}')

# %%
# Visualize some examples
def numpy_to_torch(array):
  array = jax.device_get(array)
  tensor = torch.from_numpy(array)
  tensor = tensor.permute(0, 3, 1, 2)
  return tensor

NUM_IMAGES = 8
CIFAR_images = np.stack([val_set[idx][0] for idx in range(NUM_IMAGES)], axis=0)
img_grid = torchvision.utils.make_grid(numpy_to_torch(CIFAR_images), nrow=4, normalize=True, pad_value=0.9)
img_grid = img_grid.permute(1, 2, 0)

plt.figure(figsize=(8,8))
plt.title("Image examples of the CIFAR10 dataset")
plt.imshow(img_grid)
plt.axis('off')
plt.show()
plt.close()

# %%
"""
## Initialize model
"""

# %%
def initialize_model(
    seed=42,
    patch_size=4, embed_dim=64, hidden_dim=192,
    n_heads=3, drop_p=0.1, num_layers=12, mlp_dim=768, num_classes=10
):
  main_rng = jax.random.PRNGKey(seed)
  x = jnp.ones(shape=(5, 32, 32, 3))
  # ViT
  model = ViT(
      patch_size=patch_size,
      embed_dim=embed_dim,
      hidden_dim=hidden_dim,
      n_heads=n_heads,
      drop_p=drop_p,
      num_layers=num_layers,
      mlp_dim=mlp_dim,
      num_classes=num_classes
  )
  main_rng, init_rng, drop_rng = random.split(main_rng, 3)
  params = model.init({'params': init_rng, 'dropout': drop_rng}, x, train=True)['params']
  return model, params, main_rng

# %%
vit_model, vit_params, vit_rng = initialize_model()

# %%
jax.tree_map(lambda x: x.shape, vit_params)

# %%
"""
## Define loss
"""

# %%
def calculate_loss(params, state, rng, batch, train):
  imgs, labels = batch
  rng, drop_rng = random.split(rng)
  logits = state.apply_fn({'params': params}, imgs, train=train, rngs={'dropout': drop_rng})
  loss = optax.softmax_cross_entropy_with_integer_labels(logits=logits, labels=labels).mean()
  acc = (logits.argmax(axis=-1) == labels).mean()
  return loss, (acc, rng)

# %%
"""
## Train step
"""

# %%
@jax.jit
def train_step(state, rng, batch):
  loss_fn = lambda params: calculate_loss(params, state, rng, batch, train=True)
  # Get loss, gradients for loss, and other outputs of loss function
  (loss, (acc, rng)), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params)
  # Update parameters and batch statistics
  state = state.apply_gradients(grads=grads)
  return state, rng, loss, acc

# %%
"""
## Evaluate step
"""

# %%
@jax.jit
def eval_step(state, rng, batch):
  _, (acc, rng) = calculate_loss(state.params, state, rng, batch, train=False)
  return rng, acc

# %%
logger = SummaryWriter(log_dir='vit_jax_logs')

# %%
"""
## Train function
"""

# %%
def train_epoch(train_loader, epoch_idx, state, rng):
  metrics = defaultdict(list)
  for batch in tqdm(train_loader, desc='Training', leave=False):
    state, rng, loss, acc = train_step(state, rng, batch)
    metrics['loss'].append(loss)
    metrics['acc'].append(acc)
  for key in metrics.keys():
    arg_val = np.stack(jax.device_get(metrics[key])).mean()
    logger.add_scalar('train/' + key, arg_val, global_step=epoch_idx)
    print(f'[epoch {epoch_idx}] {key}: {arg_val}')
  return state, rng

# %%
"""
## Evaluate function
"""

# %%
def eval_model(data_loader, state, rng):
  # Test model on all images of a data loader and return avg loss
  correct_class, count = 0, 0
  for batch in data_loader:
    rng, acc = eval_step(state, rng, batch)
    correct_class += acc * batch[0].shape[0]
    count += batch[0].shape[0]
  eval_acc = (correct_class / count).item()
  return eval_acc

# %%
"""
## Train model
"""

# %%
def train_model(train_loader, val_loader, state, rng, num_epochs=100):
  best_eval = 0.0
  for epoch_idx in tqdm(range(1, num_epochs + 1)):
    state, rng = train_epoch(train_loader, epoch_idx, state, rng)
    if epoch_idx % 1 == 0:
      eval_acc = eval_model(val_loader, state, rng)
      logger.add_scalar('val/acc', eval_acc, global_step=epoch_idx)
      if eval_acc >= best_eval:
        best_eval = eval_acc
        save_model(state, step=epoch_idx)
      logger.flush()
  # Evaluate after training
  test_acc = eval_model(test_loader, state, rng)
  print(f'test_acc: {test_acc}')

# %%
"""
## Create train state
"""

# %%
def create_train_state(
    model, params, learning_rate
):
  optimizer = optax.adam(learning_rate)
  return train_state.TrainState.create(
      apply_fn=model.apply,
      tx=optimizer,
      params=params
  )

# %%
"""
## Save model
"""

# %%
def save_model(state, step=0):
  checkpoints.save_checkpoint(ckpt_dir='vit_jax_logs', target=state.params, step=step, overwrite=True)

# %%
"""
## Training
"""

# %%
state = create_train_state(vit_model, vit_params, 3e-4)

# %%
train_model(train_loader, val_loader, state, vit_rng, num_epochs=100)

# %%
