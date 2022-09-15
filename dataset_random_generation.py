import os
from os import listdir
from random import sample
# import cv2
import shutil

def listdirs(rootdir):
    path = []
    for file in listdir(rootdir):
        d = os.path.join(rootdir, file)
        if os.path.isdir(d):
            path.append(d)
    path = sample((path),len(path))
    return path

def listfiles(rootdir):
    path = []
    for file in listdir(rootdir):
        if file.lower().endswith('jpg'):
            d = os.path.join(rootdir, file)
            path.append(d)
    path = sample((path),len(path))
    path = sample((path), len(path))
    return path

dataset = '/home/neural/Documents/Sabeeha/refined_data'
Train = '/home/neural/Documents/Sabeeha/yolov5/display3/Train'
Valid = '/home/neural/Documents/Sabeeha/yolov5/display3/Valid'

if not os.path.exists(Train):
    os.makedirs(Train)
if not os.path.exists(Valid):
    os.makedirs(Valid)

total = 0
dirs = listdirs(dataset)

for dir in dirs:
    total +=len(listdir(dir))/2

print(total)

file_names = []
files = dict()

for d in dirs:
    files_in_dir = listfiles(d)
    files[d] = files_in_dir
    # print(files_in_dir)

flag =True
count = 0
img_id = 0

while(flag):
         dirs = sample(dirs, len(dirs))
         print(dirs)
         if len(dirs)>0:
             try:
                dirname = dirs[0]
                files_dir = files[dirname]

                file_in_use = files_dir[0]
                print(file_in_use)
                # img = cv2.imread(file_in_use)
                # h,w,_ = img.shape
                #img = cv2.resize(img,(int(w/2),int(h/2)),interpolation = cv2.INTER_AREA)
                file_name_img = os.path.basename(dirname)+'_'+ str(count).zfill(6)+'.jpg'
                file_name_label = os.path.basename(dirname) + '_' + str(count).zfill(6) + '.txt'

                if count<total*4/5:
                    data_dir = Train
                else:
                    data_dir = Valid
                img_path = os.path.join(data_dir,file_name_img)
                label_path = os.path.join(data_dir,file_name_label)
                shutil.copy(file_in_use,img_path)
                shutil.copy(file_in_use.replace('jpg','txt'),label_path)
                files_dir.pop(0)
                count = count + 1
             except:
                dirs.pop(0)
         else:
             flag = False
print("count",count)
