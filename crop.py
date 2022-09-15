import os
#import numpy
import cv2
import glob
path = '/home/neural/Documents/Sabeeha/Dataset/train/*.jpg'
ds =  ['Mayo410g','LiteMayo','others','Mayo810g','MayoSqueezeBottle','MayoRecycledBottle']
for name in ds:
    dirname = os.path.join('/home/neural/examples/tensorflow_examples/lite/model_maker/cropped_dataset',name)
    if not os.path.exists(dirname):
        os.makedirs(dirname)
name = {'Mayo410g':'Mayo410g','LiteMayo':'LiteMayo','NoiseTrain':'others','Mayo810g':'Mayo810g','MayoRecycledBottle':'MayoRecycledBottle','MayoSqueezeBottle':'MayoSqueezeBottle'}
files = glob.glob(path)
for file in files:
    filename = os.path.basename(file)
    suffix = filename.split('_')[0]
    dirname = os.path.join('/home/neural/examples/tensorflow_examples/lite/model_maker/cropped_dataset',name[suffix])
    filename = os.path.join(dirname,filename)
    print(filename)
    img = cv2.imread(file)
    h,w,_ = img.shape
    cropped_img = img[int(h/4):(h-int(h/7.5)),0:w]
    cv2.imwrite(filename,cropped_img)
    #shutil.copy(file,dirname)