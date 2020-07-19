import os
import shutil
anno_dir = "E:/datasets/VOCdevkit/VOC2012/ImageSets/Main"
file_list = ["boat_train.txt","boat_val.txt","cat_train.txt","cat_val.txt"]

for anno_file_name in file_list:
    anno_file_path = "{}/{}".format(anno_dir,anno_file_name)
    for line in open(anno_file_path):
        name = line.strip().split(" ")[0]
        attr = line.strip().split(" ")[1]
        if attr == "-1":
            continue

boat_train = [ line.strip().split(" ")[0] for line in open("{}/{}".format(anno_dir,"boat_train.txt")) if line.strip().split(" ")[-1] != "-1"]
boat_val = [ line.strip().split(" ")[0] for line in open("{}/{}".format(anno_dir,"boat_val.txt")) if line.strip().split(" ")[-1] != "-1"]
cat_train = [ line.strip().split(" ")[0] for line in open("{}/{}".format(anno_dir,"cat_train.txt")) if line.strip().split(" ")[-1] != "-1"]
cat_val = [ line.strip().split(" ")[0] for line in open("{}/{}".format(anno_dir,"cat_val.txt")) if line.strip().split(" ")[-1] != "-1"]

print("{} {} {} {}".format(len(boat_train),len(boat_val),len(cat_train),len(cat_val)))

for i in boat_train:
    shutil.copy("E:/datasets/VOCdevkit/VOC2012/JPEGImages/{}.jpg".format(i),"cat_and_boat/train/images/")
    shutil.copy("E:/datasets/VOCdevkit/VOC2012/Annotations/{}.xml".format(i),"cat_and_boat/train/annotations/")
for i in cat_train:
    shutil.copy("E:/datasets/VOCdevkit/VOC2012/JPEGImages/{}.jpg".format(i),"cat_and_boat/train/images/")
    shutil.copy("E:/datasets/VOCdevkit/VOC2012/Annotations/{}.xml".format(i),"cat_and_boat/train/annotations/")

for i in boat_val:
    shutil.copy("E:/datasets/VOCdevkit/VOC2012/JPEGImages/{}.jpg".format(i),"cat_and_boat/val/images/")
    shutil.copy("E:/datasets/VOCdevkit/VOC2012/Annotations/{}.xml".format(i),"cat_and_boat/val/annotations/")

for i in cat_val:
    shutil.copy("E:/datasets/VOCdevkit/VOC2012/JPEGImages/{}.jpg".format(i),"cat_and_boat/val/images/")
    shutil.copy("E:/datasets/VOCdevkit/VOC2012/Annotations/{}.xml".format(i),"cat_and_boat/val/annotations/")

train_list = []
train_list.extend(boat_train)
train_list.extend(cat_train)
train_list = set(train_list)

val_list = []
val_list.extend(boat_val)
val_list.extend(cat_val)
val_list = set(val_list)

with open("cat_and_boat/trainlist.txt","w") as f:
    for i in train_list:
        f.writelines(i+'\n')

with open("cat_and_boat/vallist.txt","w") as f:
    for i in val_list:
        f.writelines(i+'\n')