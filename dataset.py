import os
import cv2
import random
import numpy as np
from patchify import patchify
from tqdm import tqdm
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import normalize
from sklearn.model_selection import KFold 
from tensorflow.keras.utils import to_categorical
from matplotlib import pyplot as plt
from skimage.transform import AffineTransform, warp
from skimage import io, img_as_ubyte
from scipy.ndimage import rotate
from skimage.exposure import match_histograms

n_classes = 6

#Define functions for each operation
#Define seed for random to keep the transformation same for image and mask

# Make sure the order of the spline interpolation is 0, default is 3. 
#With interpolation, the pixel values get messed up.
def rotation(image, seed):
    random.seed(seed)
    angle= random.randint(-45,45)
    r_img = rotate(image, angle, mode='wrap', reshape=False, order=0)
    return r_img

def h_flip(image, seed):
    hflipped_img= np.fliplr(image)
    return  hflipped_img

def v_flip(image, seed):
    vflipped_img= np.flipud(image)
    return vflipped_img

def v_transl(image, seed):
    random.seed(seed)
    n_pixels = random.randint(-128,128)
    vtranslated_img = np.roll(image, n_pixels, axis=0)
    return vtranslated_img

def h_transl(image, seed):
    random.seed(seed)
    n_pixels = random.randint(-128,128)
    htranslated_img = np.roll(image, n_pixels, axis=1)
    return htranslated_img

transformations = {'rotate': rotation,
                   'horizontal flip': h_flip, 
                   'vertical flip': v_flip,
                   'vertical shift': v_transl,
                   'horizontal shift': h_transl
                 }                #use dictionary to store names of functions 

# def get_filepaths(IMG_PATH="../datasets/FIB Tomography/images", MASK_PATH="../datasets/FIB Tomography/images"):
def get_filepaths(IMG_PATH="../datasets/FIB Tomography/images", MASK_PATH="../datasets/FIB Tomography/images"):
    impaths,mkpaths=[],[] # to store paths of images from folder
    for im in os.listdir(IMG_PATH):  # read image name from folder and append its path into "images" array     
        impaths.append(os.path.join(IMG_PATH,im))
    for msk in os.listdir(MASK_PATH):  # read image name from folder and append its path into "images" array     
        mkpaths.append(os.path.join(MASK_PATH,msk))
    impaths.sort()
    mkpaths.sort()
    print(f"{len(impaths)} Filepaths received")
    return impaths,mkpaths

def get_original_images(images,masks):
    dim = (1280,640)
    # load original images
    images = [im for im in images if ".tif" in im]
    masks = [im for im in masks if ".png" in im]
    original_images, original_masks = [],[]
    number = random.randint(0, len(images)-1)  #PIck a number to select an image & mask
    for i in tqdm(range(len(images))):
        # Load and resize images and masks
        original_images.append(cv2.resize(cv2.imread(images[i],0), dim, interpolation=cv2.INTER_NEAREST))
        original_masks.append(cv2.resize(cv2.imread(masks[i],0), dim, interpolation=cv2.INTER_NEAREST))
    print(f"Original Images: {len(original_images)}, Original masks: {len(original_masks)}") # ensure images loaded correctly
    return original_images, original_masks

def get_custom_images(images,masks):
    ref = cv2.imread("../datasets/FIB Tomography/images/SEM Image - SliceImage - 121.tif",0)# load reference
    dim = (1280,640)
    # load original images
    images = [im for im in images if ".tif" in im]
    masks = [im for im in masks if ".png" in im]
    original_images, original_masks = [],[]
    number = random.randint(0, len(images)-1)  #PIck a number to select an image & mask
    for i in tqdm(range(len(images))):
        # Load and resize images and masks
        # Match the histogram of 100nm to 500nm
        original_images.append(cv2.resize(match_histograms(cv2.imread(images[i],0), ref), dim, interpolation=cv2.INTER_NEAREST))
        original_masks.append(cv2.resize(cv2.imread(masks[i],0), dim, interpolation=cv2.INTER_NEAREST))
    print(f"Original Images: {len(original_images)}, Original masks: {len(original_masks)}") # ensure images loaded correctly
    return original_images, original_masks

def get_augmented_images(original_images, original_masks):
    images_to_generate=100 # double
    seed_for_random = 42
    i=0   # variable to iterate till images_to_generate
    aug_images, aug_masks = [],[]
    while i<=images_to_generate: 
        number = random.randint(0, len(original_images)-1)  #PIck a number to select an image & mask
        image = original_images[number]
        mask = original_masks[number]

        transformed_image = None # initialized transformed images
        transformed_mask = None

        n = 0       #variable to iterate till number of transformation to apply
        transformation_count = random.randint(1, len(transformations)) #choose random number of transformation to apply on the image

        while n <= transformation_count:
            key = random.choice(list(transformations)) #randomly choosing method to call
            seed = random.randint(1,100)  #Generate seed to supply transformation functions. 
            transformed_image = transformations[key](image, seed)
            transformed_mask = transformations[key](mask, seed)
            n = n + 1
        if(2 in np.unique(transformed_mask) or 5 in np.unique(transformed_mask)):
            aug_images.append(transformed_image)
            aug_masks.append(transformed_mask)

        i =i+1
    return aug_images,aug_masks

def patchify_dataset(original_images,original_masks,test_images,test_masks, augment=False):
    X_test,y_test,X_train,y_train = [],[],[],[]
    train_images, train_masks = original_images, original_masks
    if augment:
        print("Augmentation Enabled")
        aug_images, aug_masks = get_augmented_images(train_images,train_masks) # augment training images
        train_images, train_masks = train_images+aug_images, train_masks+aug_masks
    print("Original: ", len(train_images))
    for i in range(len(test_images)):
        p = (128,128)# desired patch size
        image_patches,mask_patches = patchify(test_images[i],(p[0],p[1]), step=p[0]), patchify(test_masks[i],(p[0],p[1]), step=p[0]) 
        X_test.append(image_patches.reshape(50, 128, 128)), y_test.append(mask_patches.reshape(50, 128, 128))
    for i in range(len(train_images)):
        p = (128,128)# desired patch size
        image_patches,mask_patches = patchify(train_images[i],(p[0],p[1]), step=p[0]), patchify(train_masks[i],(p[0],p[1]), step=p[0])
        X_train.append(image_patches.reshape(50, 128, 128)), y_train.append(mask_patches.reshape(50, 128, 128))
    print("Total: ", len(X_train)+len(X_test))
    n, p, h, w = np.array(X_train).shape
    X_train,y_train = np.array(X_train).reshape(n*p, 128, 128),np.array(y_train).reshape(n*p, 128, 128)
    
    n, p, h, w = np.array(X_test).shape
    X_test,y_test = np.array(X_test).reshape(n*p, 128, 128),np.array(y_test).reshape(n*p, 128, 128)
    return X_train,X_test,y_train,y_test

def get_trainvaltest_split(X_train,X_test,y_train,y_test):
    # encode segmenation masks
    labelencoder1, labelencoder2 = LabelEncoder(),LabelEncoder()
    n, h, w = y_train.shape
    ny,hy,wy = y_test.shape
    y_train_reshaped = y_train.reshape(-1,1)
    y_test_masks_reshaped = y_test.reshape(-1,1)
    y_train_reshaped_encoded = labelencoder1.fit_transform(y_train_reshaped)
    y_test_masks_reshaped_encoded = labelencoder2.fit_transform(y_test_masks_reshaped)
    y_train_encoded_original_shape = y_train_reshaped_encoded.reshape(n, h, w)
    y_test_encoded_original_shape = y_test_masks_reshaped_encoded.reshape(ny, hy, wy)
    # normalize dataset
    train_images = np.expand_dims(X_train, axis=3)
    test_images = np.expand_dims(X_test, axis=3)
    train_images = normalize(train_images, axis=1) # Normalize datasets
    test_images = normalize(test_images, axis=1)
    y_train_input = np.expand_dims(y_train_encoded_original_shape, axis=3)
    y_test_input = np.expand_dims(y_test_encoded_original_shape, axis=3)
    #create train/validataion splits
    k = 100
    kf = KFold(n_splits=k, shuffle=True, random_state=8)
    train_index, test_index = next(kf.split(train_images))
    X_train,y_train = train_images[train_index,:,:], y_train_input[train_index,:,:]
    X_val,y_val= train_images[test_index,:,:], y_train_input[test_index,:,:]
    X_test, y_test = test_images, y_test_input
    # create final segmenation masks
    train_masks_cat = to_categorical(y_train, num_classes=n_classes)
    y_train_cat = train_masks_cat.reshape((y_train.shape[0], y_train.shape[1], y_train.shape[2], n_classes))
    test_masks_cat = to_categorical(y_test, num_classes=n_classes)
    y_test_cat = test_masks_cat.reshape((y_test.shape[0], y_test.shape[1], y_test.shape[2], n_classes))
    val_masks_cat = to_categorical(y_val, num_classes=n_classes)
    y_val_cat = val_masks_cat.reshape((y_val.shape[0], y_val.shape[1], y_val.shape[2], n_classes))
    return X_train,X_val,X_test, y_train,y_val,y_test, y_train_cat,y_val_cat,y_test_cat

def create_dataset(number, augment=False):
    impaths,mkpaths = get_filepaths(IMG_PATH="../datasets/FIB Tomography/images", MASK_PATH="../datasets/FIB Tomography/Labels") # get paths to all files
    original_images, original_masks = get_original_images(impaths,mkpaths) # get all original images
    test_images,test_masks = [original_images[number]], [original_masks[number]]   # create train/test split
    del original_images[number] # delete test images from training dataset
    del original_masks[number]
    X_train,X_test,y_train,y_test = patchify_dataset(original_images,original_masks,test_images,test_masks, augment) # patchify
    return get_trainvaltest_split(X_train,X_test,y_train,y_test)
def create_custom_dataset(img="../datasets/100nm/Images/Aligned SEM Images/", msk="../datasets/100nm/Images/fake_labels/", number=0, augment=False):
    impaths,mkpaths = get_filepaths(IMG_PATH=img, MASK_PATH=msk) # get paths to all files
    original_images, original_masks = get_custom_images(impaths,mkpaths) # get all original images
    test_images,test_masks = [original_images[number]], [original_masks[number]]   # create train/test split
    del original_images[number] # delete test images from training dataset
    del original_masks[number]
    X_train,X_test,y_train,y_test = patchify_dataset(original_images,original_masks,test_images,test_masks, augment) # patchify
    return get_trainvaltest_split(X_train,X_test,y_train,y_test)
def create_dataset_no_patches(number):
    impaths,mkpaths = get_filepaths(IMG_PATH="../datasets/FIB Tomography/images", MASK_PATH="../datasets/FIB Tomography/Labels") # get paths to all files
    original_images, original_masks = get_original_images(impaths,mkpaths) # get all original images
    test_images,test_masks = [original_images[number]], [original_masks[number]]   # create train/test split
    del original_images[number] # delete test images from training dataset
    del original_masks[number]
    aug_images,aug_masks = get_augmented_images(original_images,original_masks)
    train_images, train_masks = original_images+aug_images, original_masks+aug_masks # combine augmented and original images
    X_train,y_train = np.array(train_images), np.array(train_masks)
    X_test,y_test = np.array(test_images),np.array(test_masks)
    return get_trainvaltest_split(X_train,X_test,y_train,y_test)
