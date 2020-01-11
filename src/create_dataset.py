from pathlib import Path

import numpy as np
import pandas as pd


h, w = 137, 236
input_d = Path('../inputs')


def main():
    df_train = pd.read_csv(input_d/'train.csv')
    train_parquet_list = [str(input_d/f'train_image_data_{i}.parquet') for i in range(4)]

    X_train_list = []
    for i in range(4):
        df_train_ = pd.read_parquet(train_parquet_list[i])
        X_train_ = df_train_.drop('image_id', axis=1).values.reshape(-1, h, w)
        X_train_ = 255 - X_train_
        X_train_list.append(X_train_)

    X_train = np.concatenate(X_train_list, axis=0)
    np.save(input_d/'X_train.npy', X_train)
    del X_train_list

    onehot0 = pd.get_dummies(df_train['grapheme_root']).values
    onehot1 = pd.get_dummies(df_train['vowel_diacritic']).values
    onehot2 = pd.get_dummies(df_train['consonant_diacritic']).values
    y_train = np.concatenate([onehot0, onehot1, onehot2], axis=1)
    np.save(input_d/'y_train.npy', y_train)


if __name__ == '__main__':
    main()
