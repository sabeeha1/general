# import os
# import shutil
# import glob
# src = glob.glob("/home/neural/Documents/Sabeeha/database/codes/*.txt")
# database_path =glob.glob( "/home/neural/Documents/Sabeeha/database/display3/Valid/*.jpg")
# # for file in src:
# #     shutil.move(file, database_path)
# #
# yes=no = 0
# # files = os.listdir(database_path)
# for file in src:
#     # txtfile = file.replace("jpg","txt")
#     # print(file)
#     # if os.path.exists(txtfile):
#     #     print("yes")
#     #     yes +=1
#     # else:
#     #     print("No")
#     #     no+=1
#     os.remove(file)
#

import cv2
import os
import numpy as np
import shutil


def Data_augmentation(img_path, num):
    img = cv2.imread(img_path)
    # print(img_path, img)
    base_dir = os.path.dirname(img_path)
    txtfile= img_path.replace('jpg','txt')

    # Decreasing brightness
    for i in [70, 40]:
        bright = np.ones(img.shape, dtype='uint8') * i
        dim_file = cv2.subtract(img, bright)
        file_name = str(num).zfill(4) + '.jpg'
        cv2.imwrite(os.path.join(base_dir, file_name), dim_file)
        label_file = os.path.join(base_dir, file_name).replace('jpg','txt')
        shutil.copy(txtfile,label_file)
        num += 1

    # Increasing brightness
    for i in [5, 10]:
        for j in [10]:
            bright = (np.ones(img.shape, dtype='uint8') * i) + j
            bright_file = cv2.add(img, bright)
            file_name = str(num).zfill(4) + '.jpg'
            cv2.imwrite(os.path.join(base_dir, file_name), bright_file)
            label_file =os.path.join(base_dir, file_name).replace('jpg', 'txt')
            shutil.copy(txtfile, label_file)
            num += 1

    # creating blurriness
    for i in [(8, 8), (11, 11)]:
        img_blurr = cv2.blur(img, i)
        file_name = str(num).zfill(4) + '.jpg'
        cv2.imwrite(os.path.join(base_dir, file_name), img_blurr)
        label_file = os.path.join(base_dir, file_name).replace('jpg','txt')
        shutil.copy(txtfile,label_file)
        num += 1
    return num


if __name__ == "__main__":
    database_path = "/home/neural/Documents/Sabeeha/database/display3/Valid"
    datafiles = os.listdir(database_path)
    print(datafiles)
    database_length = int(len(os.listdir(database_path))/2)
    num = database_length

    imgfiles = []
    for f in datafiles:
        if f.endswith('jpg'):
            imgfiles.append(f)
# print(yes,no)
    # base_dir = os.path.dirname(database_path)
    for img in imgfiles:
        imgfile = os.path.join(database_path, img)
        num = Data_augmentation(imgfile, num )
