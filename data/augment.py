import torchvision.transforms as transforms
import numpy as np
import random as r
import PIL

def shearX(img, mag):
    magRange = [0, 0.3]
    vals = np.linspace(magRange[0], magRange[-1], 10)
    if r.randint(0, 1):
        vals = -vals

    shearVal = vals[mag]

    scale = [1, 1]
    rot = np.array([[1, 0], [0, 1]])
    scale = np.array([[1.0 / scale[0], 0], [0, 1.0 / scale[1]]])
    shear_x = np.array([[1, shearVal], [0, 1]])
    shear_y = np.array([[1, 0], [0, 1]])
    tr = np.dot(np.dot(np.dot(rot, scale), shear_x), shear_y)
    return img.transform(img.size, PIL.Image.AFFINE, (tr[0, 0], tr[0, 1], 0, tr[1, 0], tr[1, 1], 0))

def rotate(img, mag):
    magRange = [0, 30]
    vals = np.linspace(magRange[0], magRange[-1], 10)
    if r.randint(0, 1):
        vals = -vals

    return img.rotate(angle = vals[mag])

def autoCon(img):
    return PIL.ImageOps.autocontrast(img)

def invert(img):
    return PIL.ImageOps.invert(img)

def equalize(img):
    return PIL.ImageOps.equalize(img)

def solarize(img, mag):
    magRange = [0, 256]
    vals = np.linspace(magRange[0], magRange[-1], 10)

    return PIL.ImageOps.solarize(img, threshold = vals[mag])

def posterize(img, mag):

    return PIL.ImageOps.posterize(img, bits = mag)

def contrast(img, mag):
    magRange = [0.1, 1.9]
    vals = np.linspace(magRange[0], magRange[-1], 10)

    return PIL.ImageEnhance.Contrast(img).enhance(vals[mag])

def color(img, mag):
    magRange = [0.1, 1.9]
    vals = np.linspace(magRange[0], magRange[-1], 10)

    return PIL.ImageEnhance.Color(img).enhance(vals[mag])

def brightness(img, mag):
    magRange = [0.1, 1.9]
    vals = np.linspace(magRange[0], magRange[-1], 10)

    return PIL.ImageEnhance.Brightness(img).enhance(vals[mag])

def sharpness(img, mag):
    magRange = [0.1, 1.9]
    vals = np.linspace(magRange[0], magRange[-1], 10)

    return PIL.ImageEnhance.Sharpness(img).enhance(vals[mag])

def subAugPolicy(img, subp):
    if subp == 0:
        if r.random() < 0.4:
            img = posterize(img, 8)
        if r.random() < 0.6:
            img = rotate(img, 9)
    elif subp == 1:
        if r.random() < 0.6:
            img = solarize(img, 5)
        if r.random() < 0.6:
            img = autoCon(img)
    elif subp == 2:
        if r.random() < 0.8:
            img = equalize(img)
        if r.random() < 0.6:
            img = equalize(img)
    elif subp == 3:
        if r.random() < 0.6:
            img = posterize(img, 7)
        if r.random() < 0.6:
            img = posterize(img, 6)
    elif subp == 4:
        if r.random() < 0.4:
            img = equalize(img)
        if r.random() < 0.2:
            img = solarize(img, 4)
    elif subp == 5:
        if r.random() < 0.4:
            img = equalize(img)
        if r.random() < 0.8:
            img = rotate(img, 8)
    elif subp == 6:
        if r.random() < 0.6:
            img = solarize(img, 3)
        if r.random() < 0.6:
            img = equalize(img)
    elif subp == 7:
        if r.random() < 0.8:
            img = posterize(img, 5)
        if r.random() <= 1.0:
            img = equalize(img)
    elif subp == 8:
        if r.random() < 0.2:
            img = rotate(img, 3)
        if r.random() < 0.6:
            img = solarize(img, 8)
    elif subp == 9:
        if r.random() < 0.6:
            img = equalize(img)
        if r.random() < 0.4:
            img = posterize(img, 6)
    elif subp == 10:
        if r.random() < 0.8:
            img = rotate(img, 8)
        if r.random() < 0.4:
            img = color(img, 0)
    elif subp == 11:
        if r.random() < 0.4:
            img = rotate(img, 9)
        if r.random() < 0.6:
            img = equalize(img)
    elif subp == 12:
        if r.random() < 0.0:
            img = equalize(img)
        if r.random() < 0.8:
            img = equalize(img)
    elif subp == 13:
        if r.random() < 0.6:
            img = invert(img)
        if r.random() <= 1.0:
            img = equalize(img)
    elif subp == 14:
        if r.random() < 0.6:
            img = color(img, 4)
        if r.random() <= 1.0:
            img = equalize(img)
    elif subp == 15:
        if r.random() < 0.8:
            img = rotate(img, 8)
        if r.random() <= 1.0:
            img = color(img, 2)
    elif subp == 16:
        if r.random() < 0.8:
            img = color(img, 8)
        if r.random() < 0.8:
            img = solarize(img, 7)
    elif subp == 17:
        if r.random() < 0.4:
            img = sharpness(img, 7)
        if r.random() < 0.6:
            img = invert(img)
    elif subp == 18:
        if r.random() < 0.6:
            img = shearX(img, 5)
        if r.random() <= 1.0:
            img = equalize(img)
    elif subp == 19:
        if r.random() < 0.4:
            img = color(img, 0)
        if r.random() < 0.6:
            img = equalize(img)
    elif subp == 20:
        if r.random() < 0.4:
            img = equalize(img)
        if r.random() < 0.2:
            img = solarize(img, 4)
    elif subp == 21:
        if r.random() < 0.6:
            img = solarize(img, 5)
        if r.random() < 0.6:
            img = autoCon(img)
    elif subp == 22:
        if r.random() < 0.6:
            img = invert(img)
        if r.random() <= 1.0:
            img = equalize(img)
    elif subp == 23:
        if r.random() < 0.6:
            img = color(img, 4)
        if r.random() <= 1.0:
            img = contrast(img, 8)
    else:
        if r.random() < 0.8:
            img = equalize(img)
        if r.random() < 0.6:
            img = equalize(img)
    return img

def applyAug(img):
    img = transforms.RandomHorizontalFlip()(img)
    subp = r.randint(0, 24)
    img = subAugPolicy(img, subp)
    return img
