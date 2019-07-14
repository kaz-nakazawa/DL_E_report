# coding: utf-8
# # keras
# ## cifar10
# #### 実行に時間がかかるため割愛となっているが、実行してみた。
# 8:14 - 08:40:04 終了:
# データセット cifar10<br>
# 32x32ピクセルのカラー画像データ<br>
# 10種のラベル「飛行機、自動車、鳥、猫、鹿、犬、蛙、馬、船、トラック」<br>
# トレーニングデータ数:50000, テストデータ数:10000<br>
# http://www.cs.toronto.edu/~kriz/cifar.html

# CIFAR-10のデータセットのインポート
from keras.datasets import cifar10
from datetime import datetime
print (datetime.now ().strftime ('%H:%M:%S') + ' 開始: ')

(x_train, d_train), (x_test, d_test) = cifar10.load_data ()

# CIFAR-10の正規化
from keras.utils import to_categorical

# 特徴量の正規化
x_train = x_train / 255.
x_test = x_test / 255.

# クラスラベルの1-hotベクトル化
d_train = to_categorical (d_train, 10)
d_test = to_categorical (d_test, 10)

# CNNの構築
import keras
from keras.models import Sequential
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.layers.core import Dense, Dropout, Activation, Flatten
import numpy as np

model = Sequential ()

model.add (Conv2D (32, (3, 3), padding='same', input_shape=x_train.shape [1:]))
model.add (Activation ('relu'))
model.add (Conv2D (32, (3, 3)))
model.add (Activation ('relu'))
model.add (MaxPooling2D (pool_size=(2, 2)))
model.add (Dropout (0.25))

model.add (Conv2D (64, (3, 3), padding='same'))
model.add (Activation ('relu'))
model.add (Conv2D (64, (3, 3)))
model.add (Activation ('relu'))
model.add (MaxPooling2D (pool_size=(2, 2)))
model.add (Dropout (0.25))

model.add (Flatten ())
model.add (Dense (512))
model.add (Activation ('relu'))
model.add (Dropout (0.5))
model.add (Dense (10))
model.add (Activation ('softmax'))

# コンパイル
model.compile (loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# 訓練
history = model.fit (x_train, d_train, epochs=20)

# モデルの保存
model.save ('./CIFAR-10.h5')

# 評価 & 評価結果出力
print (model.evaluate (x_test, d_test))
print (datetime.now ().strftime ('%H:%M:%S') + ' 終了: ')
