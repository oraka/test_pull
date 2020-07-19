
import h5py
import numpy as np
import shutil

from misc_utils.tensor_sampling_utils import sample_tensors
"""
1. 訓練された重みファイルをロードし、コピーを作成します。
まず、必要な学習済み重みを含むHDF5ファイル（ソースファイル）をロードします。
私たちの場合は、"VGG_coco_SSD_300x300_iter_400000.h5" (このレポのREADMEにダウンロードリンクがあります)で、
これはMS COCOで訓練されたオリジナルのSSD300モデルの重みです。

そして、その重みファイルのコピーを作成します。そのコピーが出力ファイル（出力先ファイル）になります。
"""

# TODO: Set the path for the source weights file you want to load.

weights_source_path = 'VGG_coco_SSD_300x300_iter_400000.h5'

# TODO: Set the path and name for the destination weights file
#       that you want to create.

weights_destination_path = 'ng_and_2classes_weight.h5'

# Make a copy of the weights file.
shutil.copy(weights_source_path, weights_destination_path)

# Load both the source weights file and the copy we made.
# We will load the original weights file in read-only mode so that we can't mess up anything.
weights_source_file = h5py.File(weights_source_path, 'r')
weights_destination_file = h5py.File(weights_destination_path)


"""
2. サブサンプルに必要な重みテンソルを見つけ出す
次に、サブサンプルに必要な重みのテンソルを正確に把握する必要があります。
上述したように、分類レイヤを除くすべてのレイヤの重みは問題ないので、それらについては何も変更する必要はありません。

では、SSD300の分類レイヤはどれでしょうか？名前は以下の通りです。
"""
classifier_names = ['conv4_3_norm_mbox_conf',
                    'fc7_mbox_conf',
                    'conv6_2_mbox_conf',
                    'conv7_2_mbox_conf',
                    'conv8_2_mbox_conf',
                    'conv9_2_mbox_conf']


"""
3. どのスライスを選ぶかを考える
以下の部分は任意です。念のため、1つの分類レイヤーを見て、何をしたいのかを説明しておきます。それが気にならなければ、次のセクションに進んでください。

どの重みのテンソルをサブサンプルにしたいかはわかりましたが、それらのテンソルのどの要素を（少なくともいくつ）残したいかを決める必要があります。
分類器層の最初の層、"conv4_3_3_norm_mbox_conf "を見てみましょう。カーネルとバイアスの2つの重みテンソルは以下の形をしています。
"""
conv4_3_norm_mbox_conf_kernel = weights_source_file[classifier_names[0]][classifier_names[0]]['kernel:0']
conv4_3_norm_mbox_conf_bias = weights_source_file[classifier_names[0]][classifier_names[0]]['bias:0']

print("Shape of the '{}' weights:".format(classifier_names[0]))
print()
print("kernel:\t", conv4_3_norm_mbox_conf_kernel.shape)
print("bias:\t", conv4_3_norm_mbox_conf_bias.shape)
"""
kernel:	 (3, 3, 512, 324)
bias:	 (324,)

だから、最後の軸は324の要素を持っています。これはなぜでしょうか？

MS COCOは80のクラスを持っていますが、モデルは1つの'backgroud'クラスも持っているので、実質的には81のクラスになります。
conv4_3_3_norm_mbox_loc' レイヤーは各空間位置に 4 つのボックスを予測するので、'conv4_3_3_norm_mbox_conf' レイヤーは
それら 4 つのボックスのそれぞれに 81 クラスのうちの 1 つを予測しなければなりません。
これが最後の軸が 4 * 81 = 324 の要素を持つ理由です。

では、このレイヤーの最後の軸には何個の要素が必要なのでしょうか？

上と同じ計算をしてみましょう。

私たちのデータセットは8つのクラスを持っていますが、私たちのモデルは「背景」クラスも持っているので、実質的には9つのクラスになります。
各空間位置にある4つのボックスのそれぞれについて、9つのクラスのうちの1つを予測する必要があります。
これは4 * 9 = 36要素になります。

これで、最後の軸に36要素を残し、他の軸はすべて変更せずに残したいことがわかりました。しかし、元の324要素のうちどの36要素が必要でしょうか？

ランダムに選ぶべきでしょうか？データセットのオブジェクトクラスがMS COCOのクラスと全く関係がない場合、それらの36個の要素をランダムに選ぶのは問題ありません
（次のセクションではこのケースについても説明します）。
しかし、私たちの特定の例では、これらの要素をランダムに選択することは無駄なことです。
MS COCOは、我々が必要とする8つのクラスを正確に含んでいるので、ランダムにサブサンプリングする代わりに、8つのクラスを予測するために訓練された要素を正確に取ります。

ここでは，MS COCOの中で注目している9つのクラスのインデックスを示します．

[0, 1, 2, 3, 4, 6, 8, 10, 12]

上記のインデックスは、MS COCOデータセットの以下のクラスを表しています。

背景', '人', '自転車', '車', 'オートバイ', 'バス', 'トラック', 'トラフィックライト', 'ストップサイン']

どのようにしてこれらの指標を見つけたのでしょうか？MSのCOCOデータセットのアノテーションで調べました。

これらは私たちが欲しいクラスですが、この順番ではありません。私たちのデータセットでは、このノートの一番上に書いてあるように、クラスはたまたま次のような順番で存在しています。

[background', 'car', 'truck', 'pedestrian', 'bicyclist', 'travic_light', 'motorcycle', 'bus', 'stop_sign']

例えば、'traffic_light' は、私たちのデータセットではクラス ID 5 ですが、SSD300 MS COCO モデルではクラス ID 10 です。したがって、実際に上記の 9 つのインデックスを選択したい順番は次のようになります。

[0, 3, 8, 1, 2, 10, 4, 6, 12]

そこで、324の要素のうち81個の要素のうち、上の9個の要素を選びたいと思います。これにより、以下の36個のインデックスが得られます。

"""

n_classes_source = 81
classes_of_interest = [0, 3, 8, 1, 2, 10, 4, 6, 12]

subsampling_indices = []
for i in range(int(324/n_classes_source)):
    indices = np.array(classes_of_interest) + i * n_classes_source
    subsampling_indices.append(indices)
subsampling_indices = list(np.concatenate(subsampling_indices))

print(subsampling_indices)


"""
バイアスベクトルとカーネルテンソルの最後の軸の両方から選びたい36の要素のインデックスです。

これは'conv4_3_3_norm_mbox_conf'レイヤーの詳細な例です。
もちろん、このレイヤーの重みを実際にはまだサブサンプリングしていません。
次のセクションのコードは、すべての分類器レイヤーのサブサンプリングを実行します。
"""


"""
4. 分類器の重みをサブサンプル
このセクションのコードは，ソース重みファイルのすべての分類器層を反復処理し，各分類器層に対して以下のステップを実行します．

ソース・ウェイト・ファイルからカーネル・テンソルとバイアス・テンソルを取得します。
最後の軸のサブサンプリング・インデックスを計算します。カーネルの最初の3軸は変更されないままです。
コピー先の重みファイルの対応するカーネルとバイアスのテンソルを、新しく作成されたサブサンプリングされたカーネルとバイアスのテンソルで上書きします。
2番目のステップでは、前のセクションで説明したことを行います。

最後の軸をサブサンプルするのではなく、アップサンプルしたい場合は、以下の変数 classes_of_interest をあなたが望む長さに設定してください。
追加された要素は、ランダムに、またはオプションでゼロで初期化されます。詳細は sample_tensors() のドキュメントをチェックしてください。

"""
# TODO: Set the number of classes in the source weights file. Note that this number must include
#       the background class, so for MS COCO's 80 classes, this must be 80 + 1 = 81.
n_classes_source = 81
# TODO: Set the indices of the classes that you want to pick for the sub-sampled weight tensors.
#       In case you would like to just randomly sample a certain number of classes, you can just set
#       `classes_of_interest` to an integer instead of the list below. Either way, don't forget to
#       include the background class. That is, if you set an integer, and you want `n` positive classes,
#       then you must set `classes_of_interest = n + 1`.
# classes_of_interest = [0, 3, 8, 1, 2, 10, 4, 6, 12]
classes_of_interest = 3 # Uncomment this in case you want to just randomly sub-sample the last axis instead of providing a list of indices.

for name in classifier_names:
    # Get the trained weights for this layer from the source HDF5 weights file.
    kernel = weights_source_file[name][name]['kernel:0'].value
    bias = weights_source_file[name][name]['bias:0'].value

    # Get the shape of the kernel. We're interested in sub-sampling
    # the last dimension, 'o'.
    height, width, in_channels, out_channels = kernel.shape
    
    # Compute the indices of the elements we want to sub-sample.
    # Keep in mind that each classification predictor layer predicts multiple
    # bounding boxes for every spatial location, so we want to sub-sample
    # the relevant classes for each of these boxes.
    if isinstance(classes_of_interest, (list, tuple)):
        subsampling_indices = []
        for i in range(int(out_channels/n_classes_source)):
            indices = np.array(classes_of_interest) + i * n_classes_source
            subsampling_indices.append(indices)
        subsampling_indices = list(np.concatenate(subsampling_indices))
    elif isinstance(classes_of_interest, int):
        subsampling_indices = int(classes_of_interest * (out_channels/n_classes_source))
    else:
        raise ValueError("`classes_of_interest` must be either an integer or a list/tuple.")
    
    # Sub-sample the kernel and bias.
    # The `sample_tensors()` function used below provides extensive
    # documentation, so don't hesitate to read it if you want to know
    # what exactly is going on here.
    new_kernel, new_bias = sample_tensors(weights_list=[kernel, bias],
                                          sampling_instructions=[height, width, in_channels, subsampling_indices],
                                          axes=[[3]], # The one bias dimension corresponds to the last kernel dimension.
                                          init=['gaussian', 'zeros'],
                                          mean=0.0,
                                          stddev=0.005)
    
    # Delete the old weights from the destination file.
    del weights_destination_file[name][name]['kernel:0']
    del weights_destination_file[name][name]['bias:0']
    # Create new datasets for the sub-sampled weights.
    weights_destination_file[name][name].create_dataset(name='kernel:0', data=new_kernel)
    weights_destination_file[name][name].create_dataset(name='bias:0', data=new_bias)

# Make sure all data is written to our output file before this sub-routine exits.
weights_destination_file.flush()

"""
これで完了です。

コピー先のウェイトファイルにある 'conv4_3_norm_mbox_conf' レイヤーのウェイトの形状を手早く調べてみましょう。
"""