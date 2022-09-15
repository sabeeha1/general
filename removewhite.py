# -*- coding: utf-8 -*-
"""
Created on Wed Jul 28 09:24:07 2021

@author: fsarwar
"""
from PIL import Image, ImageChops
import matplotlib.pyplot as plt
import os
import sys

if getattr(sys, 'frozen', False) and hasattr(sys, '_MEIPASS'):
    os.chdir(sys._MEIPASS)


# img_name = sys.argv[1]
def strTrimmedImage(cwd, img_path, img_name):
    # cwd =os.getcwd()
    # file_name = os.path.split(img_name)
    # os.makedirs(os.path.join(file_name[0],'Trimmed_test'), exist_ok=True)
    # img_path = os.path.join(file_name[0],'Trimmed_test')
    # file_name = os.path.split(img_name)
    im = Image.open(os.path.join(cwd, img_name))
    #   print('original image path, trimmed image path and file name are as follows respectively: ',img_path,img_name,im.filename)
    (width, height) = im.size
    #  bg = Image.new(im.mode, im.size, im.getpixel((0,0)))  #if there is not white space in top
    # left corner then it does not trim the image properly
    temp = Image.new(im.mode, im.size, im.getpixel((width - 1, height - 1)))
    # Computing difference between actual image and newly designed image
    diff = ImageChops.difference(im, temp)
    # diff = ImageChops.add(diff, diff, 2.0, -20)
    diff = ImageChops.add(diff, diff, 2.0, -100)  # Keep the offset smaller if the backgroup has less variation
    bbox = diff.getbbox()  # Get the bounding box of the non-zero regions in the image
    if bbox:
        imtrimmed = im.crop(bbox)
        plt.imshow(imtrimmed)
        imtrimmed.save(os.path.join(img_path, img_name))
        # imtrimmed.save(os.path.join(img_path,file_name[1]))
        # print('Image is trimmed and the path is: ',os.path.join(img_path,file_name[1]))
        # return os.path.join(img_path,file_name[1])
    else:
        print('This image cannot be trimmed any further. This function is returning a path of the main file.')
        # return os.path.join(file_name[0],file_name[1])


def convertImage(cwd, img_path, img_name):
    img = Image.open(os.path.join(cwd, img_name))
    if (img_name.endswith('.JPG') | img_name.endswith('.JPEG')):
        # print(img_name)
        name = img_name.split('.')
        img_name = name[0] + '.PNG'
        # print(img_name)
        img.save(os.path.join(cwd, img_name))
        img = Image.open(os.path.join(cwd, img_name))
        # print(img_name)

    img = img.convert("RGBA")

    datas = img.getdata()

    newData = []
    # pixel_value = img.getpixel((1,1))
    # print(pixel_value)
    for item in datas:
        check = round((item[0] + item[1] + item[2]) / 3)
        if (check > 245):
            # if item[0] == pixel_value[0] and item[1] == pixel_value[1] and item[2] == pixel_value[3]:
            newData.append((255, 255, 255, 0))
        else:
            newData.append(item)

    img.putdata(newData)
    plt.imshow(img)
    img.save(os.path.join(img_path, img_name))
    print("Successful")


# img_name = input("Enter image path: ")
# print('Image path given is:', img_name)
# img_name = sys.argv[1]
# out = strTrimmedImage(img_name)
# strTrimmed = strTrimmedImage("C:\\Users\\fsarwar\\Documents\\DataSetCreation\\Frames\\154.jpg")
# ("Path of trimmed image is: ", strTrimmed)
# input()

# img_name = input("Enter image path: ")
# print("The name is: ", img_name)


cwd = "path\\to\\PresentationImages"
os.makedirs(os.path.join(cwd, 'Transparent'), exist_ok=True)  # to create new folder in current directory
img_path = os.path.join(cwd, 'Transparent')
img_name = '1.png'
# convertImage(cwd,img_path,img_name)
# cwd =os.getcwd()
# cwd="C:\\Users\\fsarwar\\Downloads\\Cookies"
# os.makedirs(os.path.join(cwd,'Trimmed'), exist_ok=True) # to create new folder in current directory
# img_path = os.path.join(cwd,'Trimmed')
img_extensions = ['.png', '.jpeg', '.jpg', '.tiff']
listfiles = os.listdir(cwd)
listfiles = [f for f in listfiles
             if any(f.lower().endswith(ext) for ext in img_extensions)]

for index, img_name in enumerate(listfiles):
    #  im = Image.open(img_name)
    # strTrimmedImage(cwd,img_path,img_name)
    convertImage(cwd, img_path, img_name)
    # plt.figure(index)
    # plt.imshow(imtrimed)
    # imtrimed.save(os.path.join(img_path,img_name))