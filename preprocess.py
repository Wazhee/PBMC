import numpy as np
import os
from scipy import ndimage
import scipy.ndimage.morphology as morpho
# Load SAM Model
from segment_anything import SamPredictor, sam_model_registry
import cv2
from tqdm import tqdm
import matplotlib.pyplot as plt

# Specify ROI
input_point = np.array([[300, 550],[1150, 600]])
input_label = np.array([1, 1])
input_box = np.array([100, 200, 1350, 875])
#mask_input = logits[np.argmax(scores), :, :]  # Choose the model's best mask

# create and initialize directories
semA, semB = '../datasets/100nm/Images/SEM Image/', '../datasets/100nm/Images/Aligned SEM Images/'
# edsA edsB = "EDS", "Aug EDS"
if os.path.exists(semB) == False:
    os.makedirs(semB)
    
device = "cpu"
# Load SAM model
sam = sam_model_registry["vit_b"](checkpoint="sam_vit_b_01ec64.pth")
predictor = SamPredictor(sam)

"""
find x_min, x_max coordinates:

  - find where the masks begins 
  - find where the masks ends 
  - x_min and x_max used to generate final bounding box

  TIP: COULD IMPROVE RUNTIME BY USING ROI COORDINATES
"""
def find_x_coordinates(mask):
    binary_mask = mask > 0

    # Find the min and max x-coordinates of the white region
    non_zero_coords = np.argwhere(binary_mask)
    min_x, max_x = non_zero_coords[:, 1].min(), non_zero_coords[:, 1].max()

    return min_x, max_x

"""
find y_min, y_max coordinates:

  - find where the masks begins 
  - find where the masks ends 
  - y_min and y_max used to generate final bounding box

  TIP: COULD IMPROVE RUNTIME BY USING ROI COORDINATES
"""
def find_y_coordinates(mask):
    binary_mask = mask > 0
    # Find the min and max y-coordinates of the white region
    non_zero_coords = np.argwhere(binary_mask)
    min_y, max_y = non_zero_coords[:, 0].min(), non_zero_coords[:, 0].max()
    return min_y, max_y


"""
generate_mask function

predictor: makes prediction based on image and ROI
return: masks, scores, logits

masks: SAM prediction
scores: how confident is SAM in its predictions
logits: output of the neural network before activation 
        e.g. the models logic...
"""
def generate_mask(image, input_point, input_label):
    predictor.set_image(image) # image for segmentation
    return predictor.predict(
                point_coords=input_point,
                point_labels=input_label,
                box=input_box,
                multimask_output=False,
                )

"""Function for removing small objects and filling holes"""
def remove_small_objects(im,small_object_size_threshold,max_dilat):
    # detect image objects
    sz_big=10000
    sz_small=small_object_size_threshold
    dfac = int( max_dilat*(1-min(1,(max(0,sz_small)/sz_big))) )
    labeled, nr_objects = ndimage.label(im)
    result = labeled*0
    for obj_id in range(1, nr_objects+1):
        # creates a binary image with the current object
        obj_img = (labeled==obj_id)
        # computes object's area
        area = np.sum(obj_img)
        if area>small_object_size_threshold:
            if max_dilat>0:
                # dilatation factor inversely proportional to area
                dfac = int( max_dilat*(1-min(1,(max(0,area-sz_small)/sz_big))) )  
            # dilates object
                dilat = morpho.binary_dilation(obj_img, iterations=dfac)
            else:
                dilat =obj_img
            result += dilat#obj_img
            #result=np.logical_or(result,obj_img)
    return result

"""load SEM image"""
files = os.listdir(semA)
filepath = f"{semA}/{files[0]}"
blur_scale = 10

input_point = np.array([[300, 550],[1150, 600]])
input_label = np.array([1, 1])
input_box = np.array([130, 140, 1380, 860])

for i in tqdm(range(len(files))):
    # load image
    filepath = f"{semA}/{files[i]}"

    image = cv2.imread(filepath)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # convert to RGB
    
    kernel = (blur_scale, blur_scale)
    blur = cv2.blur(image,kernel) # blur image

    # # visualize ROI
    # plt.figure(figsize=(10,10))
    # plt.imshow(image); plt.title(f"{blur.shape[0]} x {blur.shape[1]}\nRegion and Points Interests")
    # show_box(input_box, plt.gca())
    # show_points(input_point, input_label, plt.gca(), marker_size=375)
    # plt.axis('off')
    # plt.show()

    # generate mask
    masks, _, _ = generate_mask(blur, # load blurred image
                                input_point, 
                                input_label)
    # get furthest xy points
    mask = np.uint8(remove_small_objects(masks[0],100,0))   # clean mask with RSO function
    # mask1,mask2 = np.array(mask),np.array(mask) # to avoid overwitting original mask
    
    # find bounding xy coordinates
    x_min, x_max = find_x_coordinates(mask)
    y_min, y_max = find_y_coordinates(mask)
    
    # Crop and resize FIB-SEM image based new xy coordinates
    cropped = image[y_min:y_max,x_min:x_max, :]
    augmented = cv2.resize(cropped, (373,195), cv2.INTER_CUBIC)
    print(x_max - x_min, len(image), print(cropped.shape))
    plt.imshow(image, cmap="gray");plt.axis(False);plt.figure()
    plt.imshow(mask,cmap="gray");plt.axis(False);plt.figure()
    plt.imshow(cropped);plt.axis(False)
    # save results
    #savepath = f"{semB}/{files[i].split('.')[0]}.png"
    savepath = f"{semB}/{files[i]}"
    cv2.imwrite(savepath, augmented)