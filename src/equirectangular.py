import random
import math

def random_on_sphere():
    cosTheta = -2.0 * random.random() + 1.0
    # sinTheta = math.sqrt(1.0 - cosTheta * cosTheta)
    phi = 2.0 * math.pi * random.random()
    theta = math.acos(cosTheta)
    return theta, phi
    # return [sinTheta * math.cos(phi), sinTheta * math.sin(phi), cosTheta]



def get_color(img, u: float, v: float):
    height, width = img.shape[:2]
    x = int(u * width)
    y = int(v * height)
    return img[y, x]
