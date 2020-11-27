import cv2
import os
import glob
f_train=open("huawei_train.odgt",'w')
f_val=open("huawei_val.odgt",'w')
train_img_paths=glob.glob("huawei_data/train/images/*")
train_label_paths=glob.glob("huawei_data/train/labels/*")
val_img_paths=glob.glob("huawei_data/val/images/*")
val_label_paths=glob.glob("huawei_data/val/labels/*")

tmp_dic={}
for i in range(len(train_img_paths)):
    img=train_img_paths[i]
    label=train_label_paths[i]
    assert img.split('/')[-1]==label.split('/')[-1]
    tmp_dic['fpath_img']=img
    tmp_dic['fpath_segm'] = label
    tmp_dic['width'] = 1024
    tmp_dic['height'] = 1024
    f_train.write(str(tmp_dic)+'\n')
f_train.close()
for i in range(len(val_img_paths)):
    img=val_img_paths[i]
    label=val_label_paths[i]
    assert img.split('/')[-1]==label.split('/')[-1]
    tmp_dic['fpath_img']=img
    tmp_dic['fpath_segm'] = label
    tmp_dic['width'] = 1024
    tmp_dic['height'] = 1024
    f_val.write(str(tmp_dic)+'\n')
f_val.close()