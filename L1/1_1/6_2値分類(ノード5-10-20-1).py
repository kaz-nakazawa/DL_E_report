import sys, os
sys.path.append(os.pardir)  # 親ディレクトリのファイルをインポートするための設定
import numpy as np
from common import functions

def print_vec(text, vec):
    print("*** " + text + " ***")
    print(vec)
    #print("shape: " + str(x.shape))
    print("")


# 2値分類
# 2-3-1ネットワーク

# ！試してみよう_ノードの構成を 5-10-20-1 に変更してみよう

# ウェイトとバイアスを設定
# ネートワークを作成
def init_network():
    print ("##### ネットワークの初期化 #####")

    network = {}
    network ['W1'] = np.array ([
        [0.1, 0.3, 0.5, 0.1, 0.3, 0.5, 0.1, 0.3, 0.5, 0.1],
        [0.1, 0.3, 0.5, 0.1, 0.3, 0.5, 0.1, 0.3, 0.5, 0.1],
        [0.1, 0.3, 0.5, 0.1, 0.3, 0.5, 0.1, 0.3, 0.5, 0.1],
        [0.1, 0.3, 0.5, 0.1, 0.3, 0.5, 0.1, 0.3, 0.5, 0.1],
        [0.1, 0.3, 0.5, 0.1, 0.3, 0.5, 0.1, 0.3, 0.5, 0.1]
    ])
    network ['W2'] = np.random.rand (10, 20)
    network ['W3'] = np.random.rand (20, 1)

    network ['b1'] = np.random.rand (10)
    network ['b2'] = np.random.rand (20)
    network ['b3'] = np.random.rand (1)

    return network


# プロセスを作成
def forward(network, x):
    print ("##### 順伝播開始 #####")

    W1, W2, W3 = network ['W1'], network ['W2'], network ['W3']
    b1, b2, b3 = network ['b1'], network ['b2'], network ['b3']

    # 隠れ層の総入力
    u1 = np.dot (x, W1) + b1
    # 隠れ層1の総出力
    z1 = functions.relu (u1)
    # 隠れ層２層への総入力
    u2 = np.dot (z1, W2) + b2
    # 隠れ層2の出力
    z2 = functions.relu (u2)

    u3 = np.dot (z2, W3) + b3
    z3 = functions.sigmoid (u3)
    y = z3
    print_vec ("総入力1", u1)
    print_vec ("中間層出力1", z1)
    print_vec ("総入力2", u2)
    print_vec ("出力1", y)
    print ("出力合計: " + str (np.sum (y)))

    return y, z1


# 入力値
x = np.array ([1., 2., 2., 4., 5.])

# 目標出力
d = np.array ([1])
network = init_network ()
y, z1 = forward (network, x)
# 誤差
loss = functions.cross_entropy_error (d, y)

## 表示
print ("\n##### 結果表示 #####")
print_vec ("中間層出力", z1)
print_vec ("出力", y)
print_vec ("訓練データ", d)
print_vec ("交差エントロピー誤差", loss)

