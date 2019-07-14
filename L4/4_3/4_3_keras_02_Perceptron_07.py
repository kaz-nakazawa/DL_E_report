# coding: utf-8
# # keras
# ## 単純パーセプトロン
# OR回路
# # ### [try]
# -  エポック数を300に変更しよう

import numpy as np
import matplotlib.pyplot as plt

# ## 単純パーセプトロン
# OR回路
#
# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# ### [try]
# -  np.random.seed(0)をnp.random.seed(1)に変更
# -  エポック数を100に変更
# -  AND回路, XOR回路に変更
# -  OR回路にしてバッチサイズを10に変更
# -  エポック数を300に変更しよう
# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# 


# モジュール読み込み
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.optimizers import SGD

# 乱数を固定値で初期化
np.random.seed (0)

# シグモイドの単純パーセプトロン作成
model = Sequential ()
model.add (Dense (input_dim=2, units=1))
model.add (Activation ('sigmoid'))
model.summary ()

model.compile (loss='binary_crossentropy', optimizer=SGD (lr=0.1))

# トレーニング用入力 X と正解データ T
X = np.array ([[0, 0], [0, 1], [1, 0], [1, 1]])
T = np.array ([[0], [1], [1], [1]])

# トレーニング
# -  エポック数を300に変更しよう
# model.fit (X, T, epochs=30, batch_size=1)
model.fit (X, T, epochs=300, batch_size=1)

# トレーニングの入力を流用して実際に分類
Y = model.predict_classes (X, batch_size=1)

print ("TEST 単純パーセプトロン OR回路 エポック数を300")
print (Y == T)
