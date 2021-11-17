import torch
import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers
from torch.utils.data import DataLoader

from networks.simclr import SimCLR
from networks.nnclr import NNCLR
import networks.resnet as resnet
from data.iNaturalist_dataset import iNaturalistDataset
import click


@click.command()
@click.option('--train-metadata-file-path', type=click.Path(), required=True,
              help='Path of training JSON metadata file')
@click.option('--validation-metadata-file-path', type=click.Path(), required=False,
              help='Path of validation JSON metadata file')
@click.option('--batch-size', type=click.INT, default=192, help="Batch size")
@click.option('--gpus', type=click.INT, default=1, help="Number of GPUs to use")
@click.option('--image-size', type=click.INT, default=224, help="Image size (one side)")
@click.option('--epochs', type=click.INT, default=100, help="Number of epochs to train for")
@click.option('--method', type=click.Choice(["simclr", "nnclr"]), default="simclr", help="Method with which to train")
@click.option('--encoder', type=click.Choice(
    ["reset18", "resnet34", "resnet50", "resnet101", "resnet152"]), default="resnet50", help="Encoder to use")
@click.option('--encoder-weights', type=click.Path(), required=False, help="Encoder weights, omit for random init")
@click.option('--encoder-key', type=click.STRING, required=False, help="Key of encoder weights in state dict")
def main(
        train_metadata_file_path,
        validation_metadata_file_path,
        batch_size,
        gpus,
        image_size,
        epochs,
        method,
        encoder,
        encoder_weights,
        encoder_key,
):
    pl.seed_everything(42)

    tb_logger = pl_loggers.TensorBoardLogger("logs/")
    trainer = pl.Trainer(
        logger=tb_logger,
        gpus=gpus,
        precision=16 if gpus > 0 else 32,
        min_epochs=epochs,
        max_epochs=epochs
    )

    train_dataset = iNaturalistDataset(train_metadata_file_path, (image_size, image_size))
    train_loaders = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=16)

    val_loaders = None
    if validation_metadata_file_path:
        val_dataset = iNaturalistDataset(validation_metadata_file_path, (image_size, image_size))
        val_loaders = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=16)

    base_encoder = getattr(resnet, encoder)(pretrained=False)
    if encoder_weights is not None:
        state_dict = torch.load(encoder_weights)["state_dict"]
        if encoder_key:
            torch.nn.modules.utils.consume_prefix_in_state_dict_if_present(state_dict, prefix=encoder_key)
        base_encoder.load_state_dict(state_dict, strict=False)
    model = None
    if method == "simclr":
        model = SimCLR(base_encoder.fc.in_features, 128, (image_size, image_size), encoder=base_encoder)
    elif method == "nnclr":
        model = NNCLR(base_encoder.fc.in_features, 128, (image_size, image_size), 32768, encoder=base_encoder)
    else:
        raise ValueError(f"Unknown method {method}")
    # explicit device for setup step
    device = torch.device('cuda:0') if gpus > 0 else torch.device('cpu')
    model.to(device)

    trainer.fit(model, train_dataloaders=train_loaders, val_dataloaders=val_loaders)


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
