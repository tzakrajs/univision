#!/usr/bin/env python3
import argparse
from io import BytesIO, IOBase
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter
from skimage.measure import compare_ssim as ssim
from x256 import x256

# Disable DeprecationWarnings, specifically because of:
#   `imread` is deprecated in SciPy 1.0.0, and will be removed in 1.2.0.
#   Use ``imageio.imread`` instead.
import warnings
warnings.simplefilter("ignore", category=DeprecationWarning)
# However, imageio.imread is incompatible with file-like object so...
from scipy.misc import imread

# Character Sets (represented as integers)
CHAR_SETS={}
CHAR_SETS['windows'] = [9600, 9604, 9608, 9612, 9616, 9617, 9618, 9619, 32]
CHAR_SETS['default'] = ([x for x in range(9602, 9615)] + # shapes
                        [x for x in range(9616, 9621)] + # more shapes
                        [x for x in range(9698, 9701)] + [32]) # triangles, blank

def main():
    """Executes CLI conversion based on arguments passed through argparse"""
    p = argparse.ArgumentParser(description='Convert images into unicode')
    p.add_argument('image', metavar='<path>', type=str,
                   help='path to the file, use - for stdin')
    p.add_argument('--no-x256', action='store_false', dest='x256', default=True,
                   help='prints with x256 unicode coloring')
    p.add_argument('--char-set', metavar='<name>', default='default',
                   help='prints with character set (e.g. windows)')
    args = p.parse_args()
    print_image_as_unicode(args.image, char_set=CHAR_SETS[args.char_set],
                           x256=args.x256)


def mse(image_a, image_b):
    """Calculate the mean squared error between two images."""
    # Credit Adrian Rosebrock
    # https://www.pyimagesearch.com/2014/09/15/python-compare-two-images/
    err = np.sum((image_a.astype("float") - image_b.astype("float")) ** 2)
    err /= float(image_a.shape[0] * image_a.shape[1])
    return err


def standardize_format(image):
    """Convert to numpy array as floats for use in comparisons"""
    if isinstance(image, Image.Image):
        new_file = BytesIO()
        image.save(new_file, format='PNG')
        new_file.seek(0)
        return imread(new_file).astype(float)
    elif isinstance(image, IOBase):
        return imread(image).astype(float)
    elif isinstance(image, np.ndarray):
        return image
    else:
        raise RuntimeError("Couldn't determine type of image to normalize.")


def compare(image_a, image_b):
    """
    Compare two image files, can be given as Image or File objects.
    Comparison returns a float which indicates the relative similarity.
    Lower is more similar.
    """
    image_a = standardize_format(image_a)
    grayscale_image_a = to_grayscale(image_a)
    image_b = standardize_format(image_b)
    grayscale_image_b = to_grayscale(image_b)
    err = mse(image_a, image_b)
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


def print_image_as_unicode(image_file, **kwargs):
    """
    Ingest a file and slice it into 10x20 bitmaps which are compared with
    bitmaps of unicode charcters. The most similar character is printed with
    x256 color which is most like the average color for the 10x20 bitmap slice.
    """
    char_set = kwargs['char_set']
    x256_mode = kwargs['x256']
    height = 20 # height of unicode character
    width = 10 # width of the unicode characters we are using
    # Credit ElTero and ABM (https://stackoverflow.com/a/7051075)
    if image_file == '-':
        source = sys.stdin.buffer
        image_file = BytesIO()
        image_file.write(source.read())
    im = Image.open(image_file)
    imgwidth, imgheight = im.size

    for row in range(imgheight//height):
        last_avg_color = np.array([0,0,0])
        for column in range(imgwidth//width):
            box = (column*width, row*height, (column+1)*width, (row+1)*height)
            cropped = im.crop(box)
            lowest_value = 100000
            lowest_unicode = None
            for unicode in char_set:
                unicode = chr(unicode)
                dissimilarity = compare(create_unicode_image(unicode), cropped)
                if dissimilarity < lowest_value:
                    lowest_value = dissimilarity
                    lowest_unicode = unicode
            if x256_mode:
                # Credit: Ruan B. (until URL)
                avg_color_per_row = np.average(cropped, axis=0)
                avg_color = np.average(avg_color_per_row, axis=0)[:3]
                x256_color = str(x256.from_rgb(*avg_color))
                # https://stackoverflow.com/a/43112217
                composite_color = np.average(np.array([avg_color,
                                                       last_avg_color]),
                                             axis=0)
                x256_bg_color = str(x256.from_rgb(*avg_color))
                if lowest_unicode == chr(32):
                    print('\033[48;5;{0}m{1}\033[0m'.format(x256_color,
                                                           chr(32)), end='')
                else:
                    print('\033[38;5;{0}m\033[48;5;{1}m'.format(x256_color,
                                                                x256_bg_color) + 
                          '{0}\033[0m'.format(lowest_unicode), end='')
                last_avg_color = avg_color
            else:
                print(lowest_unicode, end='')
        if x256_mode:
            print('\x1b[39m', end='\r\n')
        else:
            print('', end='\r\n')


if __name__ == '__main__':
    main()
