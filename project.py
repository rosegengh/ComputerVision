import cv2
import os
import exifread
from imutils import paths
import argparse
import math
from PIL import Image, ImageStat
from exif import Image


def load_images_from_folder(folder):
    images = []
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder,filename))
        if img is not None:
            images.append(img)
    return images

def variance_of_laplacian(image):
    # compute the Laplacian of the image and then return the focus measure, which is simply the variance of the Laplacian
    return cv2.Laplacian(image, cv2.CV_64F).var()

def iso_mean_pixel(im):
    stat = ImageStat.Stat(im)
    return stat.mean[0]

def iso_rms_pixel(im):
    stat = ImageStat.Stat(im)
    return stat.rms[0]

def iso_perceived(im):
    stat = ImageStat.Stat(im)
    gs = (math.sqrt(0.241*(r**2) + 0.691*(g**2) + 0.068*(b**2)) 
         for r,g,b in im.getdata())
    return sum(gs)/stat.count[0]


def main():
    images = load_images_from_folder('Test_set')
    image = images[0]
    for image in images:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        color_coverted = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # cv2.imshow('Grey scale image',gray)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        pil_image = Image.fromarray(color_coverted)
        fm = variance_of_laplacian(gray)
        print('Blurrness is')
        print(fm)
        iso = iso_perceived(pil_image)
        print('brightness is')
        print(iso)
    
    
    #print(images)

main()