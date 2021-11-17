import torch
import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers
from torch.utils.data import DataLoader

import networks.resnet as resnet
from networks.finetuning import FinetunedClassifier
from data.iNaturalist_dataset import iNaturalistDataset
import click


@click.command()
@click.option('--train-metadata-file-path', type=click.Path(), required=False,
              help='Path of training JSON metadata file')
@click.option('--validation-metadata-file-path', type=click.Path(), required=False,
              help='Path of validation JSON metadata file')
@click.option('--test-metadata-file-path', type=click.Path(), required=False,
              help='Path of validation JSON metadata file')
@click.option('--batch-size', type=click.INT, default=192, help="Batch size")
@click.option('--gpus', type=click.INT, default=1, help="Number of GPUs to use")
@click.option('--image-size', type=click.INT, default=224, help="Image size (one side)")
@click.option('--epochs', type=click.INT, default=100, help="Maximum number of epochs to train for")
@click.option('--encoder', type=click.Choice(
    ["reset18", "resnet34", "resnet50", "resnet101", "resnet152"]), default="resnet50", help="Encoder to use")
@click.option('--classifier-weights', type=click.Path(), help="Classifier checkpoint to use")
def main(
        train_metadata_file_path,
        validation_metadata_file_path,
        test_metadata_file_path,
        batch_size,
        gpus,
        image_size,
        epochs,
        encoder,
        classifier_weights,
):
    train_loaders, val_loaders = None, None
    pl.seed_everything(42)
    device = torch.device('cuda:0') if gpus > 0 else torch.device('cpu')
    callbacks = []
    tb_logger = pl_loggers.TensorBoardLogger("logs/")

    if train_metadata_file_path:
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
            callbacks.append(pl.callbacks.early_stopping.EarlyStopping(monitor="val_loss"))
            callbacks.append(pl.callbacks.ModelCheckpoint(monitor="val_loss"))
        else:
            callbacks.append(pl.callbacks.early_stopping.EarlyStopping(monitor="train_loss"))
            callbacks.append(pl.callbacks.ModelCheckpoint(monitor="train_loss"))

        cls = resnet.ResNetLightningWrapper(getattr(resnet, encoder)(pretrained=False, num_classes=len(train_dataset.categories)))

    elif classifier_weights:
        cls = resnet.ResNetLightningWrapper.load_from_checkpoint(classifier_weights)
    else:
        raise ValueError("Either provide a trained classifier or data to train on!")

    cls.to(device)

    trainer = pl.Trainer(
        logger=tb_logger,
        gpus=gpus,
        precision=16 if gpus > 0 else 32,
        max_epochs=epochs,
        callbacks=callbacks
    )

    if train_metadata_file_path:
        trainer.fit(cls, train_dataloaders=train_loaders, val_dataloaders=val_loaders)
    if test_metadata_file_path:
        test_dataset = iNaturalistDataset(test_metadata_file_path, (image_size, image_size))
        test_loaders = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=16)
        trainer.test(cls, dataloaders=test_loaders)


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
