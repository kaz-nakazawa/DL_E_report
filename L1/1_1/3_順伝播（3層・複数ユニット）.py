import sys, os
sys.path.append(os.pardir)  # 親ディレクトリのファイルをインポートするための設定
import numpy as np
from common import functions

def print_vec(text, vec):
    print("*** " + text + " ***")
    print(vec)
    #print("shape: " + str(x.shape))
    print("")


# 順伝播（3層・複数ユニット）

# ウェイトとバイアスを設定
# ネートワークを作成
def init_network():
    print ("##### ネットワークの初期化 #####")
    network = {}
    # 試してみよう
    # _各パラメータのshapeを表示
    # _ネットワークの初期値ランダム生成

    input_layer_size = 3
    hidden_layer_size_1 = 10
    hidden_layer_size_2 = 5
    output_layer_size = 4
    network ['W1'] = np.random.rand (input_layer_size, hidden_layer_size_1)
    network ['W2'] = np.random.rand (hidden_layer_size_1, hidden_layer_size_2)
    network ['W3'] = np.random.rand (hidden_layer_size_2, output_layer_size)

    network ['b1'] = np.random.rand (hidden_layer_size_1)
    network ['b2'] = np.random.rand (hidden_layer_size_2)
    network ['b3'] = np.random.rand (output_layer_size)

    print_vec ("重み1", network ['W1'])
    print_vec ("重み2", network ['W2'])
    print_vec ("重み3", network ['W3'])
    print_vec ("バイアス1", network ['b1'])
    print_vec ("バイアス2", network ['b2'])
    print_vec ("バイアス3", network ['b3'])

    return network


# プロセスを作成
# x：入力値
def forward(network, x):
    print ("##### 順伝播開始 #####")

    W1, W2, W3 = network ['W1'], network ['W2'], network ['W3']
    b1, b2, b3 = network ['b1'], network ['b2'], network ['b3']

    # 1層の総入力
    u1 = np.dot (x, W1) + b1

    # 1層の総出力
    z1 = functions.relu (u1)

    # 2層の総入力
    u2 = np.dot (z1, W2) + b2

    # 2層の総出力
    z2 = functions.relu (u2)

    # 出力層の総入力
    u3 = np.dot (z2, W3) + b3

    # 出力層の総出力
    y = u3

    print_vec ("総入力1", u1)
    print_vec ("中間層1 出力", z1)
    print_vec ("総入力2", u2)
    print_vec ("中間層2 出力", z2)
    print ("出力合計: " + str (np.sum (z1)))

    return y, z1, z2


# 入力値
x = np.array([1., 2., 4.])
print_vec ("入力", x)

# ネットワークの初期化
network = init_network ()

y, z1, z2 = forward (network, x)
print("中間層出力1", z1)
print("中間層出力2", z2)
print("出力", y)