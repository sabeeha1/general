import os
from os import listdir
from random import sample
from PIL import Image
import numpy as np

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
        d = os.path.join(rootdir, file)
        path.append(d)
    path = sample((path),len(path))
    return path

dataset = 'C:\\Users\\smehtab\\PycharmProjects\\projects\\Augmented_Frames'
bg_base_dir = 'C:\\Users\\smehtab\\PycharmProjects\\projects\\base'
NN_dir = 'C:\\Users\\smehtab\\PycharmProjects\\projects\\NN_Dir'

if not os.path.exists:
    os.mkdir(NN_dir)
bg_len = len(listdir(bg_base_dir))+1
dirpath = listdirs(dataset)

file_names = []
files = dict()
for d in dirpath:
    files_in_dir = listfiles(d)
    files[d] = files_in_dir

bg_image = 1
flag =True
count = 0
img_id = 0
base_empty_space = False

while(flag):
         dirpath = sample(dirpath, len(dirpath))
         if len(dirpath)>0:

             #try:
                if base_empty_space == False:
                    print("**")
                    if bg_image%bg_len==0:
                        bg_image = 1
                    else:
                        bg_image +=1
                    bg_img_path = os.path.join(bg_base_dir,str(bg_image)+'.jpg')
                    base_img = Image.open(bg_img_path)
                    width, height = base_img.size
                    x,y = 5,5
                    bg_image+=1
                    img_name = os.path.join(NN_dir,str(img_id).zfill(5)+'.jpg')
                    base_empty_space = True
                dirname = dirpath[0]
                files_dir = files[dirname]
                file_in_use = files_dir[0]
                patch = Image.open(file_in_use)
                base_img.paste(patch, (x, y))
                patch_width, patch_height = patch.size
                x +=patch_width+5
                y +=patch_height+5
                if x>width:
                    x=0
                    y +=370
                if y>height:
                    base_empty_space = False
                    rgb_im = base_img.convert('RGB')
                    rgb_im.save(img_name)
                    img_id +=1

                files_dir.pop(0)
                count = count + 1
             #except:
                dirpath.pop(0)
         else:
             flag = False
print("count",count)


