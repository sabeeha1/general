import os
import functools
path_name = '/home/neural/Documents/Sabeeha/PycharmProjects/yolov5/Display_dataset_4500'
dirs = sorted(os.listdir(path_name))

list1 = dirs
list2 = ['train','valid']
print(list1)
if functools.reduce(lambda x, y: x and y, map(lambda a, b: a == b, list1, list2), True):
    print("Both List are same")
else:
    print("Not same")