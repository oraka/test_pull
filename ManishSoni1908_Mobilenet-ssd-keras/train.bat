set voc_dir_path=../datasets/VOCdevkit
set weight_file=MobilenetWeights\mobilenet.h5
set epochs=50
set intial_epoch=0
set checkpoint_path=allmodel
set batch_size=8
set PYTHONPATH=E:\ManishSoni1908_Mobilenet-ssd-keras;E:\ManishSoni1908_Mobilenet-ssd-keras\models
python training\train_mobilenet_ssd.py ^
--voc_dir_path=%voc_dir_path% ^
--weight_file=%weight_file% ^
--epochs=%epochs% ^
--intial_epoch=%intial_epoch% ^
--checkpoint_path=%checkpoint_path% ^
--batch_size=%batch_size% ^