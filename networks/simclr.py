import torch
import torchvision.transforms as A
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from typing import Tuple, Optional, Union
import networks.resnet as resnet


class SimCLR(pl.LightningModule):
    def __init__(self, input_dim: int, hidden_dim: int, image_size: Tuple[int, int],
                 encoder: Union[str, nn.Module] = "resnet34", **kwargs):
        super().__init__(**kwargs)

        self.encoder = encoder
        if isinstance(encoder, str):
            self.encoder = getattr(resnet, encoder)(pretrained=False)

        self.image_size = image_size

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        # make odd kernel size 10% of image width
        kernel_size = int(round(0.1 * self.image_size[0]))
        kernel_size += 1 - (kernel_size % 2)

        self.augmentations = torch.nn.Sequential(
            A.RandomHorizontalFlip(p=0.5),
            A.RandomVerticalFlip(p=0.1),
            A.RandomResizedCrop(self.image_size),
            A.RandomApply(
                nn.ModuleList([A.ColorJitter(brightness=0.8, contrast=0.8, saturation=0.8, hue=0.2)]),
                p=0.8
            ),
            A.RandomGrayscale(p=0.2),
            A.GaussianBlur(kernel_size=kernel_size, sigma=(0.01, 2.0))
        )

        self.projection_head = nn.Sequential(
            nn.Linear(self.input_dim, self.input_dim),
            nn.BatchNorm1d(self.input_dim),
            nn.ReLU(),
            nn.Linear(self.input_dim, self.input_dim),
            nn.BatchNorm1d(self.input_dim),
            nn.ReLU(),
            nn.Linear(self.input_dim, self.hidden_dim)
        )
        self.save_hyperparameters("input_dim", "hidden_dim", "image_size")

    def forward(self, x):
        return self.encoder.encode(x)

    def step(self, batch, batch_idx, mode):
        x, _ = batch
        if x.ndim == 4:  # batch, channels, height, width
            x1, x2 = x, x
        elif x.ndim == 5:  # batch, frames, channels, height, width
            x1 = x[:, torch.randint(0, x.size(1), (1,))]
            x2 = x[:, torch.randint(0, x.size(1), (1,))]
        else:
            raise ValueError("Unrecognized shape for batch")
        x1 = self.augmentations(x1)
        x2 = self.augmentations(x2)
        x1 = self.encoder.encode(x1)
        x2 = self.encoder.encode(x2)
        z1 = self.projection_head(x1)
        z2 = self.projection_head(x2)
        loss = self.compute_contrastive_loss(z1, z2)
        self.log(f"{mode}_loss", loss)
        return {
            "loss": loss,
            "z1": z1.detach(), "z2": z2.detach(),
        }

    def training_step(self, batch, batch_idx):
        return self.step(batch, batch_idx, "train")

    def validation_step(self, batch, batch_idx):
        return self.step(batch, batch_idx, "val")

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=1e-3)
        return optimizer

    def compute_contrastive_loss(self, x1: torch.Tensor, x2: torch.Tensor, tau: float = 0.1, eps: float = 1e-8,
                                 reduction: str = 'mean') -> torch.Tensor:
        """
        Computes the SimCLR contrastive loss for x1 and x2
        The corresponding entries in x1 and x2 should be positive
        samples, and all others are considered negative
        :param x1: tensor of encoded points under transform A
        :param x2: tensor of encoded points under transform B
        :param tau: temperature value
        :param eps: epsilon for cosine similarity
        :param reduction: reduction method for loss
        :return: loss
        """
        # compute the cosine similarities for the positive examples
        exp_sim_pos = torch.exp(F.cosine_similarity(x1, x2, eps=eps) / tau)
        # l2 normalization
        x1_n = x1 / x1.norm(dim=1).clamp(min=eps)[:, None]
        x2_n = x2 / x2.norm(dim=1).clamp(min=eps)[:, None]
        # compute the cosine similarities for all examples
        exp_sim_mat = torch.exp(torch.mm(x1_n, x2_n.T) / tau)
        # note that we do not mask out the positive example
        exp_sim_x1x2 = torch.sum(exp_sim_mat, dim=0)
        # "softmax" loss
        unreduced_loss = -torch.log(exp_sim_pos / exp_sim_x1x2)

        if reduction == 'none':
            return unreduced_loss
        elif reduction == 'mean':
            return torch.mean(unreduced_loss)
        elif reduction == 'sum':
            return torch.sum(unreduced_loss)
        else:
            raise ValueError(f"Unrecognized reduction {reduction}")

    def lightly_compute_contrastive_loss(self, x1: torch.Tensor, x2: torch.Tensor, tau: float = 0.1,
                                         reduction: str = 'mean') -> torch.Tensor:
        """
        Computes the Lightly version of SimCLR contrastive loss for x1 and x2
        The corresponding entries in x1 and x2 should be positive
        samples, and all others are considered negative
        :param x1: tensor of encoded points under transform A
        :param x2: tensor of encoded points under transform B
        :param tau: temperature value
        :param eps: epsilon for cosine similarity
        :param reduction: reduction method for loss
        :return: loss
        """
        output = torch.cat((x1, x2), dim=0)
        batch_size, _ = x1.shape

        # the logits are the similarity matrix divided by the temperature
        logits = torch.einsum('nc,mc->nm', output, output) / tau
        # We need to removed the similarities of samples to themselves
        logits = logits[~torch.eye(2 * batch_size, dtype=torch.bool, device=x1.device)].view(2 * batch_size, -1)

        # The labels point from a sample in out_i to its equivalent in out_(1-i)
        labels = torch.arange(batch_size, device=x1.device, dtype=torch.long)
        labels = torch.cat([labels + batch_size - 1, labels])

        loss = F.cross_entropy(logits, labels, reduction=reduction)

        return loss
