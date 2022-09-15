import os
import shutil
from random import sample
import functools

base_dir = ""

def listdirs(rootdir):
    path = []
    for file in os.listdir(rootdir):
        d = os.path.join(rootdir, file)
        if os.path.isdir(d):
            path.append(d)
    path = sample((path), len(path))
    return path


def listfiles(rootdir):
    path = []
    for file in os.listdir(rootdir):
        d = os.path.join(rootdir, file)
        path.append(d)
    path = sample((path), len(path))
    path = sample((path), len(path))
    return path


