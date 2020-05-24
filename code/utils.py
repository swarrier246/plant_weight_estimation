'''
Utilities for the project
'''
import cv2
from PIL import Image
import numpy as np
from plantcv import plantcv as pcv
import pandas as pd



def generate_plant_mask(input_image):
    '''
    Generate a mask which segments the plant out in the input image

    Args:
        input_image: the input image
    Returns:
        mask: boolean numpy array as the same size of the input image which segments the plant
    '''

    # Get a mask based on the HSV color space
    hsv_image = rgb2hsv(input_image)
    # mask = cv2.inRange(hsv_image, (36, 25, 25), (70, 255, 255))
    mask = cv2.inRange(hsv_image, (50, 25, 25), (150, 255, 255))

    # Post process the mask
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,
                            cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5)))
    mask = cv2.morphologyEx(
        mask, cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15)))

    bool_mask = mask > 0

    return bool_mask

def generate_plant_mask_new(input_image):
    '''
    Generate a mask which segments the plant out in the input image

    Ref: https://plantcv.readthedocs.io/en/latest/vis_tutorial/

    Args:
        input_image: the input image
    Returns:
        mask: boolean numpy array as the same size of the input image which segments the plant
    '''

    # Get the saturation channel
    # hsv_image = rgb2hsv(input_image)
    # s = hsv_image[:,:,1]
    s = pcv.rgb2gray_hsv(rgb_img=input_image, channel='s')

    # threshold the saturation image
    s_thresh = s > 130

    # perform blur on the thresholding
    # s_mblur = pcv.median_blur(gray_img=s_thresh, ksize=5)
    s_cnt = pcv.median_blur(gray_img=s_thresh, ksize=15)

    # extract the LAb channels and theshold them
    b = pcv.rgb2gray_lab(rgb_img=input_image, channel='b')
    a = pcv.rgb2gray_lab(rgb_img=input_image, channel='a')

    

    a_thresh = a <= 120
    b_thresh = b >= 105

    lab_mask = np.logical_and(a_thresh, b_thresh)

    lab_cnt = pcv.median_blur(gray_img=lab_mask, ksize=15)

    # join the two thresholdes mask
    bs = np.logical_and(s_cnt, lab_cnt)

    # filling small holes
    res = np.ones(bs.shape, dtype=np.uint8)*255
    res[bs] = 0
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(15,15))
    res = cv2.morphologyEx(res,cv2.MORPH_OPEN,kernel)

    res = res==0


    return res

def get_plant_object(image, mask):
    '''
    Use the mask information to filter out the plant object
    '''

    id_objects, obj_heirachy = pcv.find_objects(image, mask)

    return id_objects, obj_heirachy




def rgb2hsv(input_image):
    '''
    Convert the input RGB image to HSV image

    Args:
        input_image: the input image as an RGB numpy array
    Returns:
        hsv_image: the input image in HSV colorspace
    '''

    hsv_image = cv2.cvtColor(input_image, cv2.COLOR_RGB2HSV)

    return hsv_image


def segment_plant_area(input_image, mask):
    '''
    Segment only plant area and remove non-plant portion of masks
    Args: 
        input_image: input image
        mask: output of generate_plant_mask_new function. Contains plant and some non-plant area
    Returns:
        Plant area mask of image
    '''
    # Threshold based on value channel from hsv image
    h, s, v = cv2.split(cv2.cvtColor(input_image, cv2.COLOR_BGR2HSV))
    img_mask = v < 160
    img_mask = img_mask.astype(np.uint8)
    mask = mask.astype(np.uint8)
    final_mask = cv2.bitwise_and(img_mask, mask)
    final_mask = final_mask.astype(np.uint8)
    
    # fill small holes
    res = np.zeros(img_mask.shape, dtype=np.uint8)*255
    res[img_mask] = 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(15,15))
    res_temp = cv2.morphologyEx(final_mask,cv2.MORPH_CLOSE, kernel)
    res = cv2.morphologyEx(res_temp, cv2.MORPH_OPEN, kernel)
    res = res==1
    res = res.astype(np.uint8)
    res_dil = cv2.dilate(res,kernel,iterations=1)
    
    return res_dil

def resize_image(image):
    '''
    Resize image
    Args: image: input image (PIL)
    '''
    new_size = image.size
    new_size = [int(x*0.25) for x in new_size]
    resized_image = image.resize(new_size, resample=Image.LANCZOS)
    img = np.asarray(resized_image)
    return img
       
    