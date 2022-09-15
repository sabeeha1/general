import glob
import shutil
import os

path = '/home/neural/Documents/tflite_model_maker/DataSetMarsGummies'
dest = '/home/neural/Documents/tflite_model_maker/DataSetMarsGummies/All'
os.makedirs((dest))

for i, dir in enumerate(os.listdir(path)):
    dir = os.path.join(path,dir)
    for file in os.listdir(dir):
        file_path = os.path.join(dir,file)
        new_file = str(i)+"_"+file
        shutil.copy(file_path,)