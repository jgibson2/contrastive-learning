import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchmetrics


class FinetunedClassifier(pl.LightningModule):
    def __init__(self, encoder, feature_dim, num_classes, num_layers=1, freeze_encoder=True, **kwargs):
        super().__init__(**kwargs)
        self.feature_dim = feature_dim
        self.num_classes = num_classes
        self.num_layers = num_layers
        self.encoder = encoder
        if freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False
        modules = []
        for i in range(num_layers - 1):
            modules.extend([nn.ReLU(),
                            nn.BatchNorm1d(feature_dim),
                            nn.Linear(feature_dim, feature_dim)])
        modules += [
            nn.ReLU(),
            nn.BatchNorm1d(feature_dim),
            nn.Linear(feature_dim, num_classes),
        ]

        self.fc = nn.Sequential(
            *modules
        )
        for m in self.fc:
            if isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight)
        self.metrics = nn.ModuleDict({
            "accuracy": torchmetrics.Accuracy(num_classes=self.num_classes, average="macro"),
            "precision": torchmetrics.Precision(num_classes=self.num_classes, average="macro"),
            "recall": torchmetrics.Recall(num_classes=self.num_classes, average="macro"),
        })
        self.save_hyperparameters("feature_dim", "num_classes", "num_layers")

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=1e-3)
        return optimizer

    def forward(self, x) -> torch.Tensor:
        f = self.encoder(x)
        f = self.fc(f)
        return f

    def step(self, batch, batch_idx, mode):
        x, y = batch
        preds = self.forward(x)
        loss = F.cross_entropy(preds, y)
        self.log(f"{mode}_loss", loss)
        for metric_name, metric in self.metrics.items():
            self.log(f"{mode}_{metric_name}", metric(preds, y), on_step=True, on_epoch=True)
        return {
            "loss": loss,
        }

    def training_step(self, batch, batch_idx):
        return self.step(batch, batch_idx, "train")

    def validation_step(self, batch, batch_idx):
        return self.step(batch, batch_idx, "val")

