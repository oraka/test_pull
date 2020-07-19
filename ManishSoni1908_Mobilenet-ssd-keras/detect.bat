set voc_dir_path=../datasets/VOCdevkit
set weight_file=MobilenetWeights\mobilenet.h5
set epochs=10
set intial_epoch=0
set checkpoint_path=2attrmodel
set batch_size=8
set PYTHONPATH=E:\ManishSoni1908_Mobilenet-ssd-keras;E:\ManishSoni1908_Mobilenet-ssd-keras\models
python inference\infer_mobilenet_ssd.py