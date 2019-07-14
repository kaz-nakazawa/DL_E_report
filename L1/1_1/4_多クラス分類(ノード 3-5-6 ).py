import sys, os
sys.path.append(os.pardir)  # 親ディレクトリのファイルをインポートするための設定
import numpy as np
from common import functions

def print_vec(text, vec):
    print("*** " + text + " ***")
    print(vec)
    #print("shape: " + str(x.shape))
    print("")


# 多クラス分類
# 2-3-4ネットワーク

# ！試してみよう_ノードの構成を3-5-6に変更してみよう

# ウェイトとバイアスを設定
# ネートワークを作成
def init_network():
    print ("##### ネットワークの初期化 #####")
    network = {}
    # 試してみよう
    # _各パラメータのshapeを表示
    # _ネットワークの初期値ランダム生成
    input_layer_size = 3
    hidden_layer_size = 5
    output_layer_size = 6

    # 試してみよう
    # _各パラメータのshapeを表示
    # _ネットワークの初期値ランダム生成
    network ['W1'] = np.random.rand (input_layer_size, hidden_layer_size)
    network ['W2'] = np.random.rand (hidden_layer_size, output_layer_size)

    network ['b1'] = np.random.rand (hidden_layer_size)
    network ['b2'] = np.random.rand (output_layer_size)

    print_vec ("重み1", network ['W1'])
    print_vec ("重み2", network ['W2'])
    print_vec ("バイアス1", network ['b1'])
    print_vec ("バイアス2", network ['b2'])

    return network


# プロセスを作成
# x：入力値
def forward(network, x):
    print ("##### 順伝播開始 #####")
    W1, W2 = network ['W1'], network ['W2']
    b1, b2 = network ['b1'], network ['b2']

    # 1層の総入力
    u1 = np.dot (x, W1) + b1

    # 1層の総出力
    z1 = functions.relu (u1)

    # 2層の総入力
    u2 = np.dot (z1, W2) + b2

    # 出力値
    y = functions.softmax (u2)

    print_vec ("総入力1", u1)
    print_vec ("中間層出力1", z1)
    print_vec ("総入力2", u2)
    print_vec ("出力1", y)
    print ("出力合計: " + str (np.sum (y)))

    return y, z1


## 事前データ
# 入力値
x = np.array ([1., 2., 3.])

# 目標出力
d = np.array ([0, 0, 0, 1, 0, 0])

# ネットワークの初期化
network = init_network ()

# 出力
y, z1 = forward (network, x)

# 誤差
loss = functions.cross_entropy_error (d, y)

## 表示
print ("\n##### 結果表示 #####")
print_vec ("出力", y)
print_vec ("訓練データ", d)
print_vec ("交差エントロピー誤差", loss)

