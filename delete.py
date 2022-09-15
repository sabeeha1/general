import os
import glob

path = glob.glob('/home/neural/Documents/Sabeeha/codes/*.txt')

for file in path:
    os.remove(file)
