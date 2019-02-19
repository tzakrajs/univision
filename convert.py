#!/usr/bin/env python3
import sys
from io import BytesIO

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter
# TODO: use imageio.imread instead of scipy.misc.imread
from scipy.misc import imread
from skimage.measure import compare_ssim as ssim
from x256 import x256

# Disable DeprecationWarnings, specifically because of:
#   `imread` is deprecated in SciPy 1.0.0, and will be removed in 1.2.0.
#   Use ``imageio.imread`` instead.
# However, imageio.imread is incompatible with Pillow Image objects so...
import warnings
warnings.simplefilter("ignore", category=DeprecationWarning)

def mse(image1, image2):
    """Calculate the mean squared error between two images."""
    # Credit Adrian Rosebrock
    # https://www.pyimagesearch.com/2014/09/15/python-compare-two-images/
    err = np.sum((image1.astype("float") - image2.astype("float")) ** 2)
    err /= float(image1.shape[0] * image1.shape[1])
    return err


def compare(file1, file2):
    """
    Compare two image files, can be given as Image or File objects.
    Comparison returns a float which indicates the relative similarity.
    Lower is more similar.
    """
    if isinstance(file1, Image.Image):
        new_file1 = BytesIO()
        file1.save(new_file1, format='PNG')
        new_file1.seek(0)
        image_float1 = imread(new_file1).astype(float)
    else:
        image_float1 = imread(file1).astype(float)
    if isinstance(file2, Image.Image):
        new_file2 = BytesIO()
        file2.save(new_file2, format='PNG')
        new_file2.seek(0)
        image_float2 = imread(new_file2).astype(float)
    else:
        image_float2 = imread(file2).astype(float)
    img1 = to_grayscale(image_float1)
    img2 = to_grayscale(image_float2)
    err = mse(img1, img2)
    return err

def to_grayscale(image):
    # Credit sastanin (https://stackoverflow.com/a/3935002)
    """Converts the given image to grayscale and returns the converted image."""
    if len(image.shape) == 3:
        return np.average(image, -1) # average the last axis (color channels)
    else:
        return image

unicode_cache = {}
def create_unicode_image(unicode_character):
    """Create bitmap from given unicode character, return image file object."""
    # Check the cache
    if unicode_character in unicode_cache.keys():
        return unicode_cache[unicode_character]
    # Initialize canvas and font parameters
    # Credit: JackNova (until URL) 
    width = 10
    height = 20
    background_color=(0,0,0)
    font_size=20
    font_color=(255,255,255)
    unicode_text = unicode_character
    im = Image.new ("RGB", (width, height), background_color )
    draw = ImageDraw.Draw ( im )
    unicode_font = ImageFont.truetype("Hack-Regular.ttf", font_size)
    draw.text ((0,0), unicode_text, font=unicode_font, fill=font_color )
    # https://stackoverflow.com/a/22612295
    # Return the image as a file object
    unicode_file = BytesIO()
    im.save(unicode_file, format='PNG')
    # Cache the charcater bitmap
    unicode_cache[unicode_character] = unicode_file
    return unicode_file


def print_image_as_unicode(infile):
    """
    Ingest a file and slice it into 10x20 bitmaps which are compared with
    bitmaps of unicode charcters. The most similar character is printed with
    x256 color which is most like the average color for the 10x20 bitmap slice.
    """
    height = 20 # height of unicode character
    width = 10 # width of the unicode characters we are using
    # Credit ElTero and ABM (https://stackoverflow.com/a/7051075)
    im = Image.open(infile)
    imgwidth, imgheight = im.size
    for row in range(imgheight//height):
        last_avg_color = np.array([0,0,0])
        for column in range(imgwidth//width):
            box = (column*width, row*height, (column+1)*width, (row+1)*height)
            cropped = im.crop(box)
            # Credit: Ruan B. (until URL)
            avg_color_per_row = np.average(cropped, axis=0)
            avg_color = np.average(avg_color_per_row, axis=0)
            x256_color = str(x256.from_rgb(*avg_color[:3]))
            # https://stackoverflow.com/a/43112217
            lowest_value = 100000
            lowest_unicode = None
            characters = []
            # experiment with different characters
            characters.extend([x for x in range(9602, 9615)]) # shapes
            characters.extend([x for x in range(9616, 9621)]) # shapes
            #characters.extend([x for x in range(9622, 9631)]) # shapes
            characters.extend([x for x in range(9698, 9701)]) # triangles
            characters.append(32) # blank space
            #characters.extend([x for x in range(11, 11632)]) 
            for unicode in characters:
                unicode = chr(unicode)
                dissimilarity = compare(create_unicode_image(unicode), cropped)
                if dissimilarity < lowest_value:
                    lowest_value = dissimilarity
                    lowest_unicode = unicode
            composite_color = np.average(np.array([avg_color, last_avg_color]),
                                         axis=0)
            x256_bg_color = str(x256.from_rgb(*avg_color[:3]))
            if lowest_unicode == chr(32):
                print('\033[48;5;{0}m{1}\033[0m'.format(x256_color,
                                                       chr(32)), end='')
            else:
                print('\033[38;5;{0}m\033[48;5;{1}m'.format(x256_color,
                                                            x256_bg_color) + 
                      '{0}\033[0m'.format(lowest_unicode), end='')
            last_avg_color = avg_color
        print('', end='\r\n\x1b[39m')


if __name__ == '__main__':
    image = open(sys.argv[1], 'rb')
    print_image_as_unicode(image)