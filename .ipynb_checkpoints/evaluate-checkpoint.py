import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
from patchify import patchify, unpatchify
from keras.metrics import MeanIoU
import pandas as pd
from tensorflow.keras.utils import to_categorical
from tqdm import tqdm
from sklearn.utils import class_weight
from keras.metrics import MeanIoU
from tqdm import tqdm
from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import EarlyStopping
from keras.losses import MeanSquaredError as mse
import tensorflow.keras.backend as K
import tensorflow as tf

image_directory, mask_directory = "dataset/augmented 465/images/", "dataset/augmented 465/masks/"

def calculate_dice(y0,y1):
    scores, avg = [], 0
    for i in range(y0.shape[-1]):
        X,y = y0[:,:,i], y1[:,:,i]
        scores.append(dice(X, y))
    return scores, round(sum(scores)/len(scores), 2)

def calculate_recall(y0,y1):
    scores, avg = [], 0
    for i in range(y0.shape[-1]):
        X,y = y0[:,:,i], y1[:,:,i]
        scores.append(recall(X, y))
    return scores, round(sum(scores)/len(scores), 2)

def calculate_iou(y0,y1):
    scores, avg = [], 0
    for i in range(y0.shape[-1]):
        X,y = y0[:,:,i], y1[:,:,i]
        scores.append(iou(X, y))
    return scores, round(sum(scores)/len(scores), 2)

def calculate_precision(y0,y1):
    scores, avg = [], 0
    for i in range(y0.shape[-1]):
        X,y = y0[:,:,i], y1[:,:,i]
        scores.append(precision(X, y))
    return scores, round(sum(scores)/len(scores), 2)

def calculate_accuracy(y0,y1):
    scores, avg = [], 0
    for i in range(y0.shape[-1]):
        X,y = y0[:,:,i], y1[:,:,i]
        scores.append(accuracy(X, y))
    return scores, round(sum(scores)/len(scores), 2)

def dice_coef(y_true, y_pred, smooth=1):
    """
    Dice = (2*|X & Y|)/ (|X|+ |Y|)
         =  2*sum(|A*B|)/(sum(A^2)+sum(B^2))
    ref: https://arxiv.org/pdf/1606.04797v1.pdf
    """
    intersection = K.sum(K.abs(y_true * y_pred), axis=-1)
    return (2 * intersection + smooth) / (K.sum(K.square(y_true),-1) + K.sum(K.square(y_pred),-1) + smooth)

def dice_coef_loss(y_true, y_pred): # weighted loss
    weights = [1.17668872, 0.3557486, 2.54948503, 0.60907073, 5.82282934, 7.49824927]
    total_loss = 0
    for i in range(len(weights)):
        total_loss += weights[i] * (1-dice_coef(y_true[None,:,:,i], y_pred[None,:,:,i]))
    return total_loss

def msedice_loss(y_true, y_pred):
    #weights = [1.17668872, 0.3557486, 2.54948503, 0.60907073, 5.82282934, 7.49824927]
    weights = [.25,.25,.25,.25,.25,.25]
    total_loss = 0
    for i in range(len(weights)):
        total_loss += weights[i] * (1-dice_coef(y_true[None,:,:,i], y_pred[None,:,:,i]))
    return (K.mean(K.square(y_true - y_pred))) + total_loss

# def msedice_loss(y_true, y_pred):
#     return (K.mean(K.square(y_true - y_pred))) + (1-dice_coef(y_true, y_pred))

def dice(groundtruth_mask, pred_mask):
    intersect = np.sum(pred_mask*groundtruth_mask)
    total_sum = np.sum(pred_mask) + np.sum(groundtruth_mask)
    dice = np.mean(2*intersect/total_sum)
    return round(dice, 3) #round up to 3 decimal places

def recall(groundtruth_mask, pred_mask):
    intersect = np.sum(pred_mask*groundtruth_mask)
    total_pixel_truth = np.sum(groundtruth_mask)
    recall = np.mean(intersect/total_pixel_truth)
    return round(recall, 3)

def precision(groundtruth_mask, pred_mask):
    intersect = np.sum(pred_mask*groundtruth_mask)
    total_pixel_pred = np.sum(pred_mask)
    precision = np.mean(intersect/total_pixel_pred)
    return round(precision, 3)

def iou(groundtruth_mask, pred_mask):
    intersect = np.sum(pred_mask*groundtruth_mask)
    union = np.sum(pred_mask) + np.sum(groundtruth_mask) - intersect
    iou = np.mean(intersect/union)
    return round(iou, 3)

def accuracy(groundtruth_mask, pred_mask):
    intersect = np.sum(pred_mask*groundtruth_mask)
    union = np.sum(pred_mask) + np.sum(groundtruth_mask) - intersect
    xor = np.sum(groundtruth_mask==pred_mask)
    acc = np.mean(xor/(union + xor - intersect))
    return round(acc, 3)

def categorical_focal_loss(gamma=2.0, alpha=0.25):
    """
    Implementation of Focal Loss from the paper in multiclass classification
    Formula:
        loss = -alpha*((1-p)^gamma)*log(p)
    Parameters:
        alpha -- the same as wighting factor in balanced cross entropy
        gamma -- focusing parameter for modulating factor (1-p)
    Default value:
        gamma -- 2.0 as mentioned in the paper
        alpha -- 0.25 as mentioned in the paper
    """
    def focal_loss(y_true, y_pred):
        # Define epsilon so that the backpropagation will not result in NaN
        # for 0 divisor case
        epsilon = K.epsilon()
        # Add the epsilon to prediction value
        #y_pred = y_pred + epsilon
        # Clip the prediction value
        y_pred = K.clip(y_pred, epsilon, 1.0-epsilon)
        # Calculate cross entropy
        cross_entropy = -y_true*K.log(y_pred)
        # Calculate weight that consists of  modulating factor and weighting factor
        weight = alpha * y_true * K.pow((1-y_pred), gamma)
        # Calculate focal loss
        loss = weight * cross_entropy
        # Sum the losses in mini_batch
        loss = K.sum(loss, axis=1)
        return loss
    
    return focal_loss

focal_loss = categorical_focal_loss(alpha=[[.15, .15, .25, .15, .15, .25]])

def combined_loss(y_true, y_pred):
    # Dice Loss
    def dice_loss(y_true, y_pred):
        smooth = 1e-5
        intersection = tf.reduce_sum(y_true * y_pred, axis=[1, 2, 3])
        union = tf.reduce_sum(y_true, axis=[1, 2, 3]) + tf.reduce_sum(y_pred, axis=[1, 2, 3])
        dice = (2. * intersection + smooth) / (union + smooth)
        return 1 - tf.reduce_mean(dice)

    # Categorical Crossentropy Loss
    cce_loss = tf.keras.losses.CategoricalCrossentropy()

    # Combine the losses
    total_loss = cce_loss(y_true, y_pred) + dice_loss(y_true, y_pred)
    return total_loss

def create_csv(scores,slicex,save_dir):
    """create and initialize save directories"""
    csv_savepath = f"../results/{save_dir}"
    if os.path.exists(csv_savepath) == False:
        os.makedirs(csv_savepath)
    df = pd.DataFrame.from_dict(scores) # convert dictionary to dataframe
    df.to_csv(f"{csv_savepath}{slicex}_slice.csv", index=False) # save dataframe as csv
    
def performance_evaluation(model, X_test, y_test, n_classes, scores, slicex, save_dir):
    fig_directory = f"../results/{save_dir}/figures/"
    if os.path.exists(fig_directory) == False:
        os.makedirs(fig_directory)
    sample_dir = f"../results/{save_dir}/samples/"
    if os.path.exists(sample_dir) == False:
        os.makedirs(sample_dir)
    
    y_pred=model.predict(X_test)  # Get Model Prediction
    y_pred_argmax=np.argmax(y_pred, axis=3) # combine prediction to single image
    pred = np.expand_dims(y_pred_argmax,axis=3)
    if(X_test.shape[1] < 256):
        img,gt,msk = unpatchify(X_test),unpatchify(y_test),unpatchify(pred) # unpatchify target and prediction
        target, predicted = to_categorical(gt, num_classes=n_classes), to_categorical(msk, num_classes=n_classes) # split classes
    else:
        img,gt,msk = X_test[0,:,:,0],y_test[0,:,:,0],pred[0,:,:,0] 
        target, predicted = y_test, pred

    dice_scores, avg_dice = calculate_dice(target,predicted) # calculate dice scores
    recall_scores, avg_recall = calculate_recall(target,predicted) # calculate recall scores
    precision_scores, avg_precision = calculate_precision(target,predicted) # calculate precision scores
    iou_scores, avg_iou = calculate_iou(target,predicted) # calculate iou scores
    acc_scores, avg_acc = calculate_accuracy(target,predicted) # calculate accuracy scores
    print(f"Accuracy: {avg_acc}, Dice: {avg_dice}, IoU: {avg_iou}, Recall: {avg_recall}, Precision: {avg_precision}")
    
    # save predictions images
    cv2.imwrite(f"{fig_directory}{slicex}.png",msk) # save prediction
    plt.figure(figsize=(12, 8)); 
    plt.subplot(231);plt.imshow(img,cmap="gray");plt.axis(False);plt.title("Input SEM Image")
    plt.subplot(232);plt.imshow(gt,cmap="gray");plt.axis(False);plt.title("Ground Truth ")
    plt.subplot(233);plt.imshow(msk,cmap="gray");plt.axis(False);plt.title("Final Prediction")
    plt.savefig(f"{sample_dir}{slicex}.png") # save samples
    
    if not bool(scores): # if dictionary is empty 
        scores["Accuracy"] = [avg_acc]
        scores["Dice"] = [avg_dice]
        scores["IoU"] = [avg_iou]
        scores["Recall"] = [avg_recall]
        scores["Precision"] = [avg_precision]
    else: 
        scores["Accuracy"].append(avg_acc)
        scores["Dice"].append(avg_dice)
        scores["IoU"].append(avg_iou)
        scores["Recall"].append(avg_recall)
        scores["Precision"].append(avg_precision)
    
    #To calculate I0U for each class...
    materials = ["other", "U-Fuel","Lanthanide", "HT9 Cladding", "PT Coating", "Pore"]  # corresponding materials
    for idx in range(len(iou_scores)):
        if materials[idx] in scores:
            scores[materials[idx]].append(iou_scores[idx])
        else:
            scores[materials[idx]] = [iou_scores[idx]]
    return scores

def unpatch(patches):
    final_image = np.zeros((256,256))
    current_x = 0 # keep track of where your current image was last placed in the y coordinate
    images = [patches[0,:,:64,0],patches[1,:,:,0]]
    for image in images:
        # add an image to the final array and increment the y coordinate
        final_image[:image.shape[0],current_x:image.shape[1]+current_x] = image
        current_x += image.shape[1]
    new_image = final_image[:128,:64+128]
    
    final_image = np.zeros((256,256))
    current_x = 0 # keep track of where your current image was last placed in the y coordinate
    images = [new_image[:,:128],patches[2,:,:,0]]
    for image in images:
        # add an image to the final array and increment the y coordinate
        final_image[:image.shape[0],current_x:image.shape[1]+current_x] = image
        current_x += image.shape[1]
    return final_image[:128,:]

def combine_patches(top,bot):
    images = [top,bot]
    final_image = np.zeros((256,256))
    current_y = 0 # keep track of where your current image was last placed in the y coordinate
    for image in images:
        # add an image to the final array and increment the y coordinate
        final_image[current_y:image.shape[0]+current_y,:image.shape[1]] = image
        current_y += image.shape[0]
    return final_image

# def unpatchify(patches):
#     top = unpatch(patches[0:4])
#     bot = unpatch(patches[6:9])
#     return combine_patches(top,bot)

def unpatchify(patches):
    image_patches = np.zeros((5, 9, 128, 256))
    i,j = 0,0
    for idx in range(len(patches)):
        image_patches[i,j,:,:] = patches[idx,:,:,0]
        j+=1
        if j == 9:
            j=0
            i+=1
            
    p = (128,256)
    tmp = np.zeros((640,1280))
    for i in range(len(image_patches)):
        count = 0
        for j in range(len(image_patches[0])):
            if(j%2 == 0):
                tmp[p[0]*i:p[0]*(i+1), p[1]*count:p[1]*(count+1)] = image_patches[i,j,:,:]
                count += 1
    return tmp