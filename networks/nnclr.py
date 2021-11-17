import torch
import torchvision.transforms as A
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from typing import Tuple, Optional, Union
import networks.resnet as resnet


class NNCLR(pl.LightningModule):
    def __init__(self, input_dim: int, hidden_dim: int, image_size: Tuple[int, int], support_set_size: int, encoder: Union[str, nn.Module] = "resnet34", **kwargs):
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

        self.prediction_head = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim // 2),
            nn.BatchNorm1d(self.hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(self.hidden_dim // 2, self.hidden_dim)
        )

        self.support_set_size = support_set_size
        self.support_set_index = 0
        support_set = torch.zeros((self.support_set_size, self.hidden_dim), device=self.device)
        self.register_buffer("support_set", support_set, persistent=False)
        self.save_hyperparameters("input_dim", "hidden_dim", "image_size", "support_set_size")

    def forward(self, x):
        return self.encoder.encode(x)

    def setup(self, stage: Optional[str] = None, eps=1e-8):
        if stage == "fit":
            with torch.no_grad():
                self.support_set_index = 0
                train_dl = self.train_dataloader()
                for x, _ in train_dl:
                    x = x.to(self.support_set.device)
                    x = self.augmentations(x)
                    x = self.encoder.encode(x)
                    x = self.projection_head(x)
                    x /= x.norm(dim=1).clamp(min=eps)[:, None]
                    if self.support_set_size < self.support_set_index + x.size(0):
                        i = self.support_set_size - self.support_set_index
                        self.support_set[self.support_set_index:] = x[:i]
                        self.support_set[:x.size(0) - i] = x[i:]
                        self.support_set_index = x.size(0) - i
                        break
                    else:
                        self.support_set[self.support_set_index:self.support_set_index + x.size(0)] = x
                        self.support_set_index += x.size(0)
                for opt in self.optimizers():
                    opt.zero_grad()

                self.support_set.detach_()

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
        p1 = self.prediction_head(z1)
        p2 = self.prediction_head(z2)
        loss = self.compute_contrastive_loss(z1, z2, p1, p2)
        self.log(f"{mode}_loss", loss)
        return {
            "loss": loss,
            "z1": z1.detach(), "z2": z2.detach(),
            "p1": p1.detach(), "p2": p2.detach()
        }

    def training_step(self, batch, batch_idx):
        return self.step(batch, batch_idx, "train")

    def training_step_end(self, train_step_outputs, eps=1e-8):
        z1 = train_step_outputs["z1"]
        z1 /= z1.norm(dim=1).clamp(min=eps)[:, None]
        if self.support_set_size < self.support_set_index + z1.size(0):
            i = self.support_set_size - self.support_set_index
            self.support_set[self.support_set_index:] = z1[:i]
            self.support_set[:z1.size(0)-i] = z1[i:]
            self.support_set_index = z1.size(0) - i
        else:
            self.support_set[self.support_set_index:self.support_set_index+z1.size(0)] = z1
            self.support_set_index += z1.size(0)

    def validation_step(self, batch, batch_idx):
        return self.step(batch, batch_idx, "val")

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=1e-3)
        return optimizer

    def compute_contrastive_loss(self, z1: torch.Tensor, z2: torch.Tensor, p1: torch.Tensor, p2: torch.Tensor, tau: float = 0.1, eps: float = 1e-8,
                                 reduction: str = 'mean') -> torch.Tensor:
        """
        Computes the NNCLR contrastive loss for x1 and x2
        The corresponding entries in x1 and x2 should be positive
        samples, and all others are considered negative
        Note that we do not implement the prediction MLP from z+ -> p+
        :param x1: tensor of encoded points under transform A
        :param x2: tensor of encoded points under transform B
        :param tau: temperature value
        :param eps: epsilon for cosine similarity
        :param reduction: reduction method for loss
        :return: loss
        """
        # compute the pairwise distance matrix for each encoding
        z1_n = z1 / z1.norm(dim=1).clamp(min=eps)[:, None]
        # z2_n = z2 / z2.norm(dim=1).clamp(min=eps)[:, None]
        support_dists_x1 = torch.cdist(self.support_set, z1_n)
        # support_dists_x2 = torch.cdist(self.support_set, z2_n)
        # compute nearest neighbors
        z1_nn = self.support_set[torch.argmin(support_dists_x1, dim=0)]
        # z2_nn = self.support_set[torch.argmin(support_dists_x2, dim=0)]
        # compute the cosine similarities for the positive examples' nearest neighbors
        exp_sim_pos_nn_1 = torch.exp(F.cosine_similarity(z1_nn, p2, eps=eps) / tau)
        # exp_sim_pos_nn_2 = torch.exp(F.cosine_similarity(p1, z2_nn, eps=eps) / tau)
        # l2 normalization
        # p1_n = p1 / p1.norm(dim=1).clamp(min=eps)[:, None]
        p2_n = p2 / p2.norm(dim=1).clamp(min=eps)[:, None]
        z1_nn_n = z1_nn / z1_nn.norm(dim=1).clamp(min=eps)[:, None]
        # z2_nn_n = z2_nn / z2_nn.norm(dim=1).clamp(min=eps)[:, None]
        # compute the cosine similarities for all examples
        exp_sim_mat_nn_1 = torch.exp(torch.mm(z1_nn_n, p2_n.T) / tau)
        # exp_sim_mat_nn_2 = torch.exp(torch.mm(z2_nn_n, p1_n.T) / tau)
        # note that we do not mask out the positive example
        exp_sim_z1_nnz2 = torch.sum(exp_sim_mat_nn_1, dim=0)
        # exp_sim_z2z1_nn = torch.sum(exp_sim_mat_nn_1, dim=1)
        # exp_sim_z1z2_nn = torch.sum(exp_sim_mat_nn_2, dim=0)
        # exp_sim_z2_nnz1 = torch.sum(exp_sim_mat_nn_2, dim=1)
        # "softmax" loss, note that we make it symmetric
        unreduced_loss = -torch.log(exp_sim_pos_nn_1 / exp_sim_z1_nnz2)
        # unreduced_loss -= torch.log(exp_sim_pos_nn_1 / exp_sim_z2z1_nn)
        # unreduced_loss -= torch.log(exp_sim_pos_nn_2 / exp_sim_z1z2_nn)
        # unreduced_loss -= torch.log(exp_sim_pos_nn_2 / exp_sim_z2_nnz1)

        if reduction == 'none':
            return unreduced_loss
        elif reduction == 'mean':
            return torch.mean(unreduced_loss)
        elif reduction == 'sum':
            return torch.sum(unreduced_loss)
        else:
            raise ValueError(f"Unrecognized reduction {reduction}")
