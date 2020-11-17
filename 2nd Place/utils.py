import numpy as np
import cv2
def create_dummy_image(width, height, color=(255, 255, 255)):
    """Create dummy image"""
    image = np.zeros((height, width, 3), np.uint8)
    image[:] = color
    return image

def create_dummy_train(n,width,height,dest):
    """create dummy train images"""
    train_fnames = [f'train{i}.tif' for i in range(n)]
    train_tiles = [f'train{i}_0.jpeg' for i in range(n)]
    image = create_dummy_image(width, height)
    for i in range(n):
      cv2.imwrite(f'{dest}/{train_tiles[i]}.jpeg', image,[int(cv2.IMWRITE_JPEG_QUALITY), 70])
    return train_fnames, train_tiles