from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm


h, w = 137, 236
input_d = Path('../inputs')
output_fname = 'X_train_128x128.npy'

HEIGHT = 137
WIDTH = 236
SIZE = 128


def main():
    df_train = pd.read_csv(input_d/'train.csv')
    train_parquet_list = [
        str(input_d/f'train_image_data_{i}.parquet') for i in range(4)]

    X_train_list = []
    for i in range(4):
        df_train_ = pd.read_parquet(train_parquet_list[i])
        X_train = df_train_.drop('image_id', axis=1).values.reshape(-1, h, w)
        X_train = 255 - X_train
        X_train = crop_resize(X_train)
        X_train_list.append(X_train)
        print(f'Done {train_parquet_list[i]}')

    X_train = np.concatenate(X_train_list, axis=0)
    np.save(input_d/output_fname, X_train)
    del X_train_list

    onehot0 = pd.get_dummies(df_train['grapheme_root']).values
    onehot1 = pd.get_dummies(df_train['vowel_diacritic']).values
    onehot2 = pd.get_dummies(df_train['consonant_diacritic']).values
    y_train = np.concatenate([onehot0, onehot1, onehot2], axis=1)
    np.save(input_d/'y_train.npy', y_train)


def create_cropped_dataset(split_name='train'):
    parquet_list = [
        str(input_d/f'{split_name}_image_data_{i}.parquet') for i in range(4)]
    X_list = []
    n_samples = 0
    for dataset_i in range(len(parquet_list)):
        df = pd.read_parquet(parquet_list[dataset_i])
        n_samples += len(df)
        for sample_i in tqdm(range(len(df))):
            img = 255 - df.iloc[sample_i, 1:].values.reshape(HEIGHT, WIDTH).astype(np.uint8)  # noqa
            img = crop_resize(img)
            X_list.append(img)
    X = np.stack(X_list, axis=0)
    assert len(X) == n_samples
    np.save(input_d/f'X_{split_name}_128x128.npy', X)


def bbox(img):
    rows = np.any(img, axis=1)
    cols = np.any(img, axis=0)
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    return rmin, rmax, cmin, cmax


def crop_resize(img0, size=SIZE, pad=16):
    # crop a box around pixels large than the threshold
    # some images contain line at the sides
    ymin, ymax, xmin, xmax = bbox(img0[5:-5, 5:-5] > 80)
    # cropping may cut too much, so we need to add it back
    xmin = xmin - 13 if (xmin > 13) else 0
    ymin = ymin - 10 if (ymin > 10) else 0
    xmax = xmax + 13 if (xmax < WIDTH - 13) else WIDTH
    ymax = ymax + 10 if (ymax < HEIGHT - 10) else HEIGHT
    img = img0[ymin:ymax, xmin:xmax]
    # remove lo intensity pixels as noise
    img[img < 28] = 0
    lx, ly = xmax-xmin, ymax-ymin
    l = max(lx, ly) + pad  # TODO # noqa
    # make sure that the aspect ratio is kept in rescaling
    img = np.pad(img, [((l-ly)//2,), ((l-lx)//2,)], mode='constant')
    return cv2.resize(img, (size, size))


if __name__ == '__main__':
    # main()
    create_cropped_dataset('train')
