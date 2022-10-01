import tensorflow as tf
from tensorflow import keras
import numpy as np
import os

def main():
    epoch = 1000  # epoch数

    # データの作成
    # 入力用データ
    input_data = np.array(
        (
            [0, 0, 0],
            [0, 0, 1],
            [0, 1, 0],
            [0, 1, 1],
            [1, 0, 0],
            [1, 0, 1],
            [1, 1, 0],
            [1, 1, 1],
        ),
        dtype=np.float32,
    )
    # ラベル (教師データ)
    label_data = np.array([0, 1, 1, 2, 1, 2, 2, 3], dtype=np.int32)
    train_data, train_label = input_data, label_data  # 訓練データ
    validation_data, validation_label = input_data, label_data  # 検証データ
    # ネットワークの登録
    model = keras.Sequential(
        [
            keras.layers.Dense(6, activation='relu'),
            keras.layers.Dense(6, activation='relu'),
            keras.layers.Dense(4, activation='softmax'),
        ]
    )
    # model = keras.models.load_model(os.path.join('result', 'outmodel')) # modelのロード

    # 学習のためのmodelの設定
    model.compile(
        optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy']
    )

    # TensorBoard用の設定
    tb_cb = keras.callbacks.TensorBoard(log_dir='log', histogram_freq=1)

    # 学習の実行
    model.fit(
        train_data,#入力データ
        train_label,#ラベル
        epochs=epoch,#エポック数
        batch_size=8,#バッチサイズ
        callbacks=[tb_cb],#TosorBoardの設定
        validation_data=(validation_data, validation_label),#検証用
    )
    model.save(os.path.join('result', 'outmodel'))  # モデルの保存

if __name__ == '__main__':
    main()
