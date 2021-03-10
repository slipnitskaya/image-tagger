import base64

import tqdm
import msgpack
import argparse

import numpy as np
import pandas as pd

import torch
import torch.utils.data as td
import torch.utils.mobile_optimizer

import torchvision as tv

from typing import Dict, Tuple


@torch.no_grad()
def create_model(quantize: bool, path_to_model: str) -> torch.nn.Module:
    """
    Create pre-trained model and save it as TorchScript
    """
    if quantize:
        net = tv.models.quantization.mobilenet_v2(pretrained=True, quantize=quantize)
    else:
        net = tv.models.mobilenet_v2(pretrained=True)

    net.classifier = torch.nn.Identity()
    net.eval()

    sample_input = torch.rand(3, 256, 256).unsqueeze(0)
    traced_script_model = torch.jit.trace(net, sample_input)
    model = torch.utils.mobile_optimizer.optimize_for_mobile(traced_script_model)
    model.save(path_to_model)

    return net


@torch.no_grad()
def generate_embeddings(ds: tv.datasets.ImageFolder, net: torch.nn.Module,
                        quantize: bool, path_to_embeddings: str) -> np.ndarray:
    """
    Generate image embeddings and save them to a binary file
    """
    device = torch.device('cpu' if quantize else 'cuda')
    net = net.to(device)

    in_features = net(torch.rand(3, 256, 256, device=device).unsqueeze(0)).shape[-1]
    embeddings = np.zeros((len(ds), in_features + 1), dtype=np.float32)
    for idx, (x, y) in enumerate(tqdm.tqdm(ds)):
        emb = net(x.to(device).unsqueeze(0)).cpu().squeeze()
        embeddings[idx] = np.concatenate((emb.numpy(), np.array([y])), axis=0)

    np.save(path_to_embeddings, embeddings)

    return embeddings


def extract_statistics(embeddings: np.ndarray, idx_to_wnidx: Dict[int, str]) -> np.ndarray:
    """
    Extract class statistics from embeddings
    """
    num_classes = len(idx_to_wnidx)
    emb_len = embeddings.shape[-1] - 1
    emb_stats = np.zeros((num_classes, emb_len + emb_len + 1), dtype=np.float32)

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
        cnt = float(entry[-1].item())
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


def parse_arguments() -> Tuple[str, str, str, str, str, bool]:
    """
    Parse command line arguments
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--path-to-data', type=str)
    parser.add_argument('-w', '--path-to-words', type=str)
    parser.add_argument('-m', '--path-to-model', type=str)
    parser.add_argument('-e', '--path-to-embeddings', type=str)
    parser.add_argument('-t', '--path-to-table', type=str)
    parser.add_argument('-q', '--quantize', action='store_true', default=False)

    args = parser.parse_args()

    return args.path_to_data, args.path_to_words, args.path_to_model, args.path_to_embeddings, args.path_to_table, args.quantize


def main():
    path_to_data, path_to_words, path_to_model, path_to_embeddings, path_to_table, quantize = parse_arguments()

    print(f'Loading data...\n\tImages: {path_to_data}\n\tClass labels mapping: {path_to_words}')
    # load and transform images
    ds = tv.datasets.ImageFolder(
        root=path_to_data,
        transform=tv.transforms.Compose([
            tv.transforms.Resize(256),
            tv.transforms.CenterCrop(256),
            tv.transforms.ToTensor(),
            tv.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    )
    idx_to_wnidx = {idx: label for label, idx in ds.class_to_idx.items()}

    # map WordNet ids to labels
    wnidx_to_label = dict()
    for row_idx, row in pd.read_csv(path_to_words, sep='\t', names=['wn_idx', 'labels'], na_filter=False).iterrows():
        wn_idx = row['wn_idx']
        labels = row['labels']
        wnidx_to_label[wn_idx] = labels.split(',').pop(0).strip(' ')

    print(f'Loading model...\n\tModel: {path_to_model}')
    net = create_model(quantize, path_to_model)

    print(f'Generating embeddings...\n\tEmbeddings: {path_to_embeddings}')
    embeddings = generate_embeddings(ds, net, quantize, path_to_embeddings)

    print('Extracting statistics...')
    emb_stats = extract_statistics(embeddings, idx_to_wnidx)

    print(f'Saving data...\n\tTable: {path_to_table}')
    save_statistics(emb_stats, idx_to_wnidx, wnidx_to_label, path_to_table)


if __name__ == '__main__':
    main()
