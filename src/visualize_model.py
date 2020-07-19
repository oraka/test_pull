from keras.utils import plot_model

def to_image(model,to_file='model.png', show_shapes=True, show_layer_names=True,  rankdir='TB',dpi=96 ):
    """
    to_file: 保存する画像のファイルパス
    show_shapes: Trueなら各レイヤーのshapeを記載する
    show_layer_names: Trueなら各レイヤーのレイヤー名を記載する.
    rankdir: `rankdir` argument passed to PyDot,
    a string specifying the format of the plot:
    'TB' creates a vertical plot;
    'LR' creates a horizontal plot.
    expand_nested: whether to expand nested models into clusters.
    dpi: dot DPI.
    """
    plot_model(model, to_file='model.png',show_shapes=True, show_layer_names=True, rankdir='TB')

from keras import backend as K
from keras_loss_function.keras_ssd_loss import SSDLoss
from keras_layers.keras_layer_AnchorBoxes import AnchorBoxes
from keras_layers.keras_layer_L2Normalization import L2Normalization

def load_model_from_h5(model_path):
    
    # We need to create an SSDLoss object in order to pass that to the model loader.
    ssd_loss = SSDLoss(neg_pos_ratio=3, alpha=1.0)

    K.clear_session() # Clear previous models from memory.

    model = load_model(model_path, custom_objects={'AnchorBoxes': AnchorBoxes,
                                               'L2Normalization': L2Normalization,
                                               'compute_loss': ssd_loss.compute_loss})
    return model

if __name__ == "__main__":
    import sys
    model = load_model_from_h5(sys.argv[1])
    to_image(model)