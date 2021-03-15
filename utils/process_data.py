import os
import base64
import argparse

import tqdm
import msgpack

import numpy as np
import pandas as pd

import torch
import torch.utils.mobile_optimizer

import torchvision as tv

from typing import Dict

from constants import NUM_CHANNELS
from constants import INPUT_HEIGHT
from constants import INPUT_WIDTH
from constants import SHORT_SIDE
from constants import IMAGENET_MEAN
from constants import IMAGENET_STD

DEVICE = 'cpu'


def create_model(model: str = 'mobilenet_v2', quantize: bool = False, device: str = 'cpu') -> torch.nn.Module:
    """
    Create pre-trained model and save it as TorchScript.
    Supported architectures: mobilenet_v2, resnet18, resnet50, resnext101_32x8d, googlenet, inception_v3.
    """
    kwargs = {'pretrained': True}
    if quantize:
        base_module = tv.models.quantization
        kwargs['quantize'] = True
    else:
        base_module = tv.models

    net = getattr(base_module, model)(**kwargs)

    if hasattr(net, 'fc'):
        net.fc = torch.nn.Identity()
    elif hasattr(net, 'classifier'):
        net.classifier = torch.nn.Identity()
    else:
        raise ValueError('output layer not found')

    net = net.to(device)
    net.eval()

    sample_input = torch.rand(NUM_CHANNELS, INPUT_HEIGHT, INPUT_WIDTH, device=device).unsqueeze(0)
    traced_script_model = torch.jit.trace(net, sample_input)
    model = torch.utils.mobile_optimizer.optimize_for_mobile(
        traced_script_model, backend='Vulkan' if device == 'cuda' else device.upper())

    return model


def generate_embeddings(ds: tv.datasets.ImageFolder, net: torch.nn.Module) -> np.ndarray:
    """
    Generate image embeddings and save them to a binary file
    """
    net = net.to(DEVICE)

    in_features = net(torch.rand(NUM_CHANNELS, INPUT_HEIGHT, INPUT_WIDTH, device=DEVICE).unsqueeze(0)).shape[-1]
    embeddings = np.zeros((len(ds), in_features + 1), dtype=np.float32)
    for idx, (x, y) in enumerate(tqdm.tqdm(ds)):
        emb = net(x.to(DEVICE).unsqueeze(0)).cpu().squeeze()
        embeddings[idx] = np.concatenate((emb.detach().numpy(), np.array([y])), axis=0)

    return embeddings


def extract_statistics(embeddings: np.ndarray, idx_to_wnidx: Dict[int, str]) -> np.ndarray:
    """
    Extract class statistics from embeddings
    """
    num_classes = len(idx_to_wnidx)
    emb_len = embeddings.shape[-1] - 1
    emb_stats = np.zeros((num_classes, 2 * emb_len + 1), dtype=np.float32)

    for c in sorted(idx_to_wnidx.keys()):
        c = int(c)
        subset = embeddings[embeddings[:, -1].astype(int) == c][:, :-1]
        emb_stats[c] = np.concatenate((np.mean(subset, axis=0), np.std(subset, axis=0), np.array(subset.shape[:1])))

    return emb_stats


def save_statistics(emb_stats: np.ndarray, idx_to_wnidx: Dict[int, str],
                    wnidx_to_label: Dict[str, str], path_to_table: str) -> None:
    """
    Export statistics into a CSV table
    """
    emb_size = (emb_stats.shape[1] - 1) // 2

    rows_to_insert = list()
    for cid, entry in enumerate(emb_stats):
        label = wnidx_to_label[idx_to_wnidx[cid]]
        mean = [float(v) for v in entry[:emb_size]]
        std = [float(v) for v in entry[emb_size:2 * emb_size]]
        cnt = int(entry[-1].item())
        row = (
            cid,
            label,
            base64.b64encode(msgpack.packb(mean)).decode('utf-8'),
            base64.b64encode(msgpack.packb(std)).decode('utf-8'),
            cnt
        )
        rows_to_insert.append(row)

    with open(path_to_table, 'w+') as out_csv:
        for row in rows_to_insert:
            line = '{};{};{};{};{}\n'.format(*row)
            out_csv.write(line)


def prepare_data(path_to_data: str, train: bool = True) -> tv.datasets.ImageFolder:
    """
    Load and transform images
    """
    ds_root = os.path.join(path_to_data, 'train' if train else 'val')

    if not train:
        path_to_all_images = os.path.join(ds_root, 'images')
        ds_index = pd.read_csv(os.path.join(ds_root, 'val_annotations.txt'), sep='\t', header=None)

        for row_idx, row in tqdm.tqdm(ds_index.iterrows(), total=len(ds_index)):
            fn, wnid = row[:2]
            path_to_class = os.path.join(ds_root, wnid)
            path_to_image = os.path.join(path_to_class, fn)

            if not os.path.exists(path_to_image):
                os.makedirs(path_to_class, exist_ok=True)
                try:
                    os.rename(os.path.join(path_to_all_images, fn), path_to_image)
                except OSError:
                    pass

        if os.path.exists(path_to_all_images) and not os.listdir(path_to_all_images):
            os.rmdir(path_to_all_images)

    ds = tv.datasets.ImageFolder(
        root=ds_root,
        transform=tv.transforms.Compose([
            tv.transforms.Resize(SHORT_SIDE),
            tv.transforms.CenterCrop(SHORT_SIDE),
            tv.transforms.ToTensor(),
            tv.transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)
        ])
    )

    return ds


def extract_wordnet_labels(path_to_data: str) -> Dict:
    """
    Map WordNet ids to labels
    """
    path_to_words = os.path.join(path_to_data, 'words.txt')

    if os.path.exists(path_to_words):
        wnidx_to_label = dict()

        for row_idx, row in pd.read_csv(
                path_to_words,
                sep='\t',
                names=['wn_idx', 'labels'],
                na_filter=False
        ).iterrows():
            wn_idx = row['wn_idx']
            labels = row['labels']
            wnidx_to_label[wn_idx] = labels.split(',').pop(0).strip(' ')

    else:
        raise FileNotFoundError(path_to_words)

    return wnidx_to_label


def parse_arguments() -> argparse.Namespace:
    """
    Parse command line arguments
    """
    parser = argparse.ArgumentParser()

    parser.add_argument('-d', '--path-to-data', type=str, required=True)
    parser.add_argument('-m', '--model-arch', type=str, required=True)
    parser.add_argument('-q', '--quantize', action='store_true', default=False)
    parser.add_argument('-v', '--val-split', action='store_true', default=False)

    return parser.parse_args()


def main():
    global DEVICE

    args = parse_arguments()

    if args.quantize:
        DEVICE = 'cpu'
        suffix = '_q'
    else:
        suffix = ''

    if args.val_split:
        split = 'val'
    else:
        split = 'train'

    path_to_embeddings = os.path.join('generated', f'embeddings_{args.model_arch}_{split}{suffix}')
    path_to_statistics = os.path.join('generated', f'statistics_{args.model_arch}_{split}{suffix}')

    wnidx_to_label = extract_wordnet_labels(args.path_to_data)

    path_to_table = os.path.join('generated', f'image_tagger_{args.model_arch}_{split}{suffix}.csv')

    print(f'Loading model...\n\tModel: {args.model_arch}{suffix}')
    net = create_model(args.model_arch, args.quantize, DEVICE)

    print(f'Loading {split} data...\n\tImages: {args.path_to_data}')
    ds = prepare_data(args.path_to_data, not args.val_split)

    idx_to_wnidx = {idx: label for label, idx in ds.class_to_idx.items()}

    print(f'Generating {split} embeddings...\n\tEmbeddings: {path_to_embeddings}')
    embeddings = generate_embeddings(ds, net)
    np.save(path_to_embeddings, embeddings)

    print(f'Extracting {split} statistics...')
    emb_stats = extract_statistics(embeddings, idx_to_wnidx)
    np.save(path_to_statistics, emb_stats)

    print(f'Saving {split} data...\n\tTable: {path_to_table}')
    save_statistics(emb_stats, idx_to_wnidx, wnidx_to_label, path_to_table)


if __name__ == '__main__':
    if torch.cuda.is_available():
        DEVICE = 'cuda'

    main()
