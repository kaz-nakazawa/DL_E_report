import sys, os
sys.path.append(os.pardir)  # 親ディレクトリのファイルをインポートするための設定
import numpy as np
from common import functions

def print_vec(text, vec):
    print("*** " + text + " ***")
    print(vec)
    #print("shape: " + str(x.shape))
    print("")

# 順伝播（単層・複数ユニット）

# 重み

## 試してみよう_配列の初期化
#W = np.zeros((4,3))
#W = np.ones((4,3))
W = np.random.rand(4,3)
#W = np.random.randint(5, size=(4,3))
print_vec("重み", W)

# バイアス
b = np.array([0.1, 0.2, 0.3])
print_vec("バイアス", b)

# 入力値
x = np.array([1.0, 5.0, 2.0, -1.0])
print_vec("入力", x)

#  総入力
u = np.dot(x, W) + b
print_vec("総入力", u)

# 中間層出力 = 出力層出力
z = functions.sigmoid(u)
print_vec("中間層出力", z)
