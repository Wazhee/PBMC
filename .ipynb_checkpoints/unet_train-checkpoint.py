import tensorflow as tf
import warnings
warnings.filterwarnings('ignore')
warnings.simplefilter('ignore')

from keras.models import Model
from keras.layers import Add, Input, Conv2D, MaxPooling2D, UpSampling2D, concatenate, Conv2DTranspose,\
                            BatchNormalization, Dropout, Lambda, Activation
from keras.activations import relu
import numpy as np
from matplotlib import pyplot as plt
from skimage.transform import AffineTransform, warp
from skimage import io, img_as_ubyte
import random
import os
from scipy.ndimage import rotate
import cv2
from patchify import patchify
from tqdm import tqdm
from tqdm import tqdm


# UNet model
def multi_unet_model(n_classes=6, IMG_HEIGHT=256, IMG_WIDTH=256, IMG_CHANNELS=1):
#Build the model
    inputs = Input((IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS))
    #s = Lambda(lambda x: x / 255)(inputs)   #No need for this if we normalize our inputs beforehand
    s = inputs

    #Contraction path
    c1 = Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(s)
    c1 = Dropout(0.1)(c1)
    c1 = Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c1)
    p1 = MaxPooling2D((2, 2))(c1)
    
    c2 = Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p1)
    c2 = Dropout(0.1)(c2)
    c2 = Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c2)
    p2 = MaxPooling2D((2, 2))(c2)
     
    c3 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p2)
    c3 = Dropout(0.2)(c3)
    c3 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c3)
    p3 = MaxPooling2D((2, 2))(c3)
     
    c4 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p3)
    c4 = Dropout(0.2)(c4)
    c4 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c4)
    p4 = MaxPooling2D(pool_size=(2, 2))(c4)
     
    c5 = Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p4)
    c5 = Dropout(0.3)(c5)
    c5 = Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c5)
    
    #Expansive path 
    u6 = Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(c5)
    u6 = concatenate([u6, c4])
    c6 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u6)
    c6 = Dropout(0.2)(c6)
    c6 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c6)
     
    u7 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(c6)
    u7 = concatenate([u7, c3])
    c7 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u7)
    c7 = Dropout(0.2)(c7)
    c7 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c7)
     
    u8 = Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(c7)
    u8 = concatenate([u8, c2])
    c8 = Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u8)
    c8 = Dropout(0.1)(c8)
    c8 = Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c8)
     
    u9 = Conv2DTranspose(16, (2, 2), strides=(2, 2), padding='same')(c8)
    u9 = concatenate([u9, c1], axis=3)
    c9 = Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u9)
    c9 = Dropout(0.1)(c9)
    c9 = Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c9)
     
    outputs = Conv2D(n_classes, (1, 1), activation='softmax')(c9)
     
    model = Model(inputs=[inputs], outputs=[outputs])
    
    #NOTE: Compile the model in the main program to make it easy to test with various loss functions
    #model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    
    #model.summary()
    
    return model



# #Define functions for each operation
# #Define seed for random to keep the transformation same for image and mask

# # Make sure the order of the spline interpolation is 0, default is 3. 
# #With interpolation, the pixel values get messed up.
# def rotation(image, seed):
#     random.seed(seed)
#     angle= random.randint(-45,45)
#     r_img = rotate(image, angle, mode='wrap', reshape=False, order=0)
#     return r_img

# def h_flip(image, seed):
#     hflipped_img= np.fliplr(image)
#     return  hflipped_img

# def v_flip(image, seed):
#     vflipped_img= np.flipud(image)
#     return vflipped_img

# def v_transl(image, seed):
#     random.seed(seed)
#     n_pixels = random.randint(-128,128)
#     vtranslated_img = np.roll(image, n_pixels, axis=0)
#     return vtranslated_img

# def h_transl(image, seed):
#     random.seed(seed)
#     n_pixels = random.randint(-128,128)
#     htranslated_img = np.roll(image, n_pixels, axis=1)
#     return htranslated_img




# def get_filepaths(IMG_PATH="../datasets/FIB Tomography/images", MASK_PATH="../datasets/FIB Tomography/images"):
#     impaths,mkpaths=[],[] # to store paths of images from folder
#     for im in os.listdir(IMG_PATH):  # read image name from folder and append its path into "images" array     
#         impaths.append(os.path.join(IMG_PATH,im))
#     for msk in os.listdir(MASK_PATH):  # read image name from folder and append its path into "images" array     
#         mkpaths.append(os.path.join(MASK_PATH,msk))
#     impaths.sort()
#     mkpaths.sort()
#     print(f"{len(impaths)} Filepaths received")
#     return impaths,mkpaths


# idx = 0
# gpus = tf.config.list_physical_devices('GPU')
# if len(gpus) > 0: 
#     tf.config.experimental.set_visible_devices(gpus[idx], 'GPU')

    
    

# if __name__ == "__main__":
    
# # Leave-one-out-cross-validation experiment

# for idx in tqdm(range(1)):
#     X_train,X_val,X_test,y_train,y_val,y_test,y_train_cat,y_val_cat,y_test_cat = create_dataset(idx)
#     # start training
#     scores, iou_list, acc_list = {},[],[]
#     # initialize model
#     model = get_model()
#     model.compile(optimizer='adam', loss=LOSS, metrics=['accuracy'])  

#     early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True) # implement early_stopping mechanism
#     history = model.fit(X_train, y_train_cat, 
#                         batch_size = BATCHSIZE, 
#                         verbose=2, 
#                         epochs=EPOCHS, 
#                         validation_data=(X_val, y_val_cat), 
#                         callbacks=[early_stopping],
#                         #class_weight=class_weights,
#                         shuffle=False)
    
#     scores = performance_evaluation(model, X_test, y_test, n_classes, scores,idx) # calculate results
#     # save results of K-fold validation
#     create_csv(scores, idx)