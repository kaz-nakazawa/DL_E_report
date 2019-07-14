import sys, os
sys.path.append(os.pardir)  # 親ディレクトリのファイルをインポートするための設定
import numpy as np
from common import functions

def print_vec(text, vec):
    print("*** " + text + " ***")
    print(vec)
    #print("shape: " + str(x.shape))
    print("")


# 回帰
# 2-3-2ネットワーク

# ！試してみよう_ノードの構成を 3-5-4 に変更してみよう

# ウェイトとバイアスを設定
# ウェイトとバイアスを設定
# ネートワークを作成
def init_network():
    print ("##### ネットワークの初期化 #####")
    network = {}
    input_layer_size = 3
    hidden_layer_size = 5
    output_layer_size = 4

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
def forward(network, x):
    print ("##### 順伝播開始 #####")

    W1, W2 = network ['W1'], network ['W2']
    b1, b2 = network ['b1'], network ['b2']
    # 隠れ層の総入力
    u1 = np.dot (x, W1) + b1
    # 隠れ層の総出力
    z1 = functions.relu (u1)
    # 出力層の総入力
    u2 = np.dot (z1, W2) + b2
    # 出力層の総出力
    y = u2

    print_vec ("総入力1", u1)
    print_vec ("中間層出力1", z1)
    print_vec ("総入力2", u2)
    print_vec ("出力1", y)
    print ("出力合計: " + str (np.sum (y)))

    return y, z1


# 入力値
x = np.array ([1., 2., 3.])
network = init_network ()
y, z1 = forward (network, x)
# 目標出力
d = np.array ([2., 4.,3,6])
# 誤差
loss = functions.mean_squared_error (d, y)

## 表示
print ("\n##### 結果表示 #####")
print_vec ("中間層出力", z1)
print_vec ("出力", y)
print_vec ("訓練データ", d)
print_vec ("二乗誤差", loss)