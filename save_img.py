
# Importing Image module from PIL package
from PIL import Image
import PIL

# creating a image object (main image)
im1 = Image.open("C:\\Users\\smehtab\\PycharmProjects\\projects\\Augmented_Frames\\01\\0001.jpg")

# save a image using extension
im1 = im1.save("geeks.jpg")