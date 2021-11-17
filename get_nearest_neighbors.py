import pytorch_lightning as pl
import torch
import torchvision
from torch.utils.data import DataLoader
import pathlib

import networks.resnet as resnet
import torchvision.io as io
import torchvision.transforms.functional as AF
import torch.nn.functional as F
from networks.simclr import SimCLR
from networks.nnclr import NNCLR
from data.iNaturalist_dataset import iNaturalistDataset
import click
import tqdm


def read_image(img_path, image_size=(224, 224)):
    img = io.read_image(str(img_path.absolute()))
    img = AF.convert_image_dtype(img, torch.float)
    # take a center crop of the correct aspect ratio and resize
    crop_size = [img.shape[1], int(round(img.shape[1] * image_size[0] / image_size[1]))]
    img = AF.center_crop(img, crop_size)
    img = AF.resize(img, image_size)
    return img


@click.command()
@click.option('--metadata-file-path', type=click.Path(),
              help='Path of JSON metadata file')
@click.option('--query-image-path', type=click.Path(),
              help='Path of query image file')
@click.option('--batch-size', type=click.INT, default=192, help="Batch size")
@click.option('--image-size', type=click.INT, default=224, help="Image size (one side)")
@click.option('--method', type=click.Choice(["simclr", "nnclr"]), required=True, help="Method used to train encoder")
@click.option('--encoder-weights', type=click.Path(), required=True, help="Encoder checkpoint to use")
@click.option('--top-k', type=click.INT, default=31, help="Number of results to return")
@click.option('--output-dir', type=click.Path(), default="./results", help="Path of output dir")
def main(
        metadata_file_path,
        query_image_path,
        batch_size,
        image_size,
        method,
        encoder_weights,
        top_k,
        output_dir
):
    pl.seed_everything(42)
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    if method == "simclr":
        model = SimCLR.load_from_checkpoint(encoder_weights)
    elif method == "nnclr":
        model = NNCLR.load_from_checkpoint(encoder_weights)
    else:
        raise ValueError(f"Unknown method {method}")
    model.to(device)
    model.eval()

    query_image_path = pathlib.Path(query_image_path)
    output_dir = pathlib.Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    query_img = read_image(query_image_path, image_size=(image_size, image_size)).to(device)
    dataset = iNaturalistDataset(metadata_file_path, (image_size, image_size))
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=16,
        pin_memory=True)
    x1 = model(query_img.unsqueeze(0)).squeeze().detach()
    closest_imgs = torch.rand((top_k, *query_img.shape), device=device)
    closest_embeddings = torch.randn((top_k, *x1.shape), device=device)
    for imgs, _ in tqdm.tqdm(loader):
        imgs = imgs.to(device)
        x2 = torch.cat((closest_embeddings, model(imgs).detach()), 0)
        imgs = torch.cat((closest_imgs, imgs), 0)
        dists = F.pairwise_distance(x2, x1.expand(x2.shape[0], *x1.shape))
        topk_indices = torch.topk(dists, top_k, largest=False, sorted=True).indices
        closest_imgs = imgs[topk_indices]
        closest_embeddings = x2[topk_indices]
    img_path = output_dir / f"{query_image_path.stem}_closest_{top_k}.jpg"
    torchvision.utils.save_image(torch.cat((query_img.unsqueeze(0), closest_imgs), 0), str(img_path.absolute()))


if __name__ == "__main__":
    main()
