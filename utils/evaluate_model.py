import os
import argparse

import tqdm
import numpy as np
import pandas as pd

import torchvision as tv

from sklearn.metrics import accuracy_score
from sklearn.metrics import top_k_accuracy_score

from typing import Optional

from constants import SHORT_SIDE
from constants import IMAGENET_MEAN
from constants import IMAGENET_STD


class GaussianNaiveBayes(object):

    def __init__(self):
        super(GaussianNaiveBayes, self).__init__()

        self.labels: Optional[np.ndarray] = None
        self.num_classes: Optional[int] = None
        self.priors: Optional[np.ndarray] = None
        self.mu: Optional[np.ndarray] = None
        self.sigma: Optional[np.ndarray] = None

    def fit(self, X, y):
        self.labels, counts = np.unique(y, return_counts=True)

        num_features = X.shape[1]
        self.num_classes = len(self.labels)

        counts = counts[self.labels.argsort()]
        self.priors = counts.astype(np.float64) / counts.sum()

        self.mu = np.zeros((self.num_classes, num_features))
        self.sigma = np.zeros((self.num_classes, num_features))

        for cid in tqdm.tqdm(self.labels):
            X_cid = X[y == cid]
            self.mu[cid] = np.mean(X_cid, axis=0)
            self.sigma[cid] = np.std(X_cid, axis=0) ** 2

    def _joint_log_likelihood(self, X):
        num_samples = X.shape[0]

        joint_log_likelihood = np.zeros((num_samples, self.num_classes))
        for cid in tqdm.tqdm(self.labels):
            prior_cid = np.where(self.priors[cid] > 0.0, np.log(self.priors[cid]), -120.0)
            mu_cid = self.mu[cid]
            sigma_cid = self.sigma[cid]

            x = X - mu_cid
            posterior = -0.5 * (np.sum(np.log(2.0 * np.pi * sigma_cid)) + np.sum((x ** 2) / sigma_cid, 1))
            joint_log_likelihood[:, cid] = prior_cid + posterior

        return joint_log_likelihood

    def predict(self, X):
        return np.argmax(self._joint_log_likelihood(X), axis=1)

    def predict_likelihood(self, X):
        return self._joint_log_likelihood(X)


def prepare_data(path_to_data: str) -> tv.datasets.ImageFolder:
    """
    Load and transform images
    """
    ds_root = os.path.join(path_to_data, 'val')
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


def parse_arguments() -> argparse.Namespace:
    """
    Parse command line arguments
    """
    parser = argparse.ArgumentParser()

    parser.add_argument('-m', '--model-arch', type=str, required=True)
    parser.add_argument('-q', '--quantize', action='store_true', default=False)
    parser.add_argument('-i', '--input-dir', type=str, required=True)
    parser.add_argument('-o', '--out-dir', type=str, required=True)

    return parser.parse_args()


def main():
    args = parse_arguments()
    path_to_csv = os.path.join(args.out_dir, 'perf_clf.csv')

    if args.quantize:
        suffix = '_q'
    else:
        suffix = ''

    model_name = f'model_{args.model_arch}.pt'

    embeddings_train = np.load(os.path.join(args.input_dir, f'embeddings_{args.model_arch}_train{suffix}.npy'))
    embeddings_val = np.load(os.path.join(args.input_dir, f'embeddings_{args.model_arch}_val{suffix}.npy'))

    X_train = embeddings_train[:, :-1]
    y_train = embeddings_train[:, -1].astype(np.int64)
    X_val = embeddings_val[:, :-1]
    y_val = embeddings_val[:, -1].astype(np.int64)

    clf = GaussianNaiveBayes()
    clf.fit(X_train, y_train)

    y_jll_train = clf.predict_likelihood(X_train)
    y_jll_val = clf.predict_likelihood(X_val)

    acc_train = accuracy_score(y_train, y_jll_train.argmax(-1))
    acc_val = accuracy_score(y_val, y_jll_val.argmax(-1))

    print(model_name)
    print('\tAccuracy@1: {:.2%}'.format(top_k_accuracy_score(y_val, y_jll_val, k=1)))
    print('\tAccuracy@5: {:.2%}'.format(top_k_accuracy_score(y_val, y_jll_val, k=5)))
    print('\tAccuracy@10: {:.2%}'.format(top_k_accuracy_score(y_val, y_jll_val, k=10)))

    header = ''
    mode = 'a'
    if not os.path.exists(path_to_csv):
        header += 'architecture;train_acc;val_acc'
        mode += '+'

    with open(path_to_csv, mode) as csv:
        if header:
            csv.write(f'{header}\n')

        result = f'{args.model_arch}{suffix};{acc_train};{acc_val}'
        csv.write(f'{result}\n')


if __name__ == '__main__':
    main()
