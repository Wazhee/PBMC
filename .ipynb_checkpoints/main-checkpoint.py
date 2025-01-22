import argparse
from models import multi_unet_model, attention_unet, residual_unet
from tqdm import tqdm
import tensorflow as tf
from evaluate import *
from dataset import *

parser = argparse.ArgumentParser()
parser.add_argument('-train', action='store_true')
parser.add_argument('-model', default='unet', choices=['unet', 'attention', 'residual'])
parser.add_argument('-dataset', default='fib', choices=['fib'])
parser.add_argument('-augment', action='store_true') # enable data augmentation
parser.add_argument('-early_stopping', action='store_true') # enable early_stopping
parser.add_argument('-gpu', type=int, default=0, choices=[0, 1, 2, 3])

args = parser.parse_args()
dataset = args.dataset
early_stopping = args.early_stopping
model = args.model

gpus = tf.config.list_physical_devices('GPU')
if len(gpus) > 0: 
    tf.config.experimental.set_visible_devices(gpus[args.gpu], 'GPU')


def training_loop():
# Leave-one-out-cross-validation experiment
    for idx in tqdm(range(65)):
        X_train,X_val,X_test,y_train,y_val,y_test,y_train_cat,y_val_cat,y_test_cat = create_dataset(idx)
        print(f"X_train.shape: {X_train.shape}, X_val.shape: {X_val.shape}, X_test.shape: {X_test.shape}")
        IMG_HEIGHT,IMG_WIDTH,IMG_CHANNELS = X_train.shape[1],X_train.shape[2],X_train.shape[3]
        def get_model():
            return multi_unet_model(n_classes=n_classes, IMG_HEIGHT=IMG_HEIGHT, IMG_WIDTH=IMG_WIDTH, IMG_CHANNELS=IMG_CHANNELS)
        # start training
        scores, iou_list, acc_list = {},[],[]
        # initialize model
        model = get_model()
        model.compile(optimizer='adam', loss=LOSS, metrics=['accuracy'])  

        early_stopping = EarlyStopping(monitor='val_loss', patience=30, restore_best_weights=True) # implement early_stopping mechanism
        history = model.fit(X_train, y_train_cat, 
                            batch_size = BATCHSIZE, 
                            verbose=2, 
                            epochs=EPOCHS, 
                            validation_data=(X_val, y_val_cat), 
                            callbacks=[early_stopping],
                            #class_weight=class_weights,
                            shuffle=True)

        scores = performance_evaluation(model, X_test, y_test, n_classes, scores,idx) # calculate results
    #     # save results of K-fold validation
        create_csv(scores, idx)
        
def train_unet(n_classes, IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS ):
    def get_model():
        return multi_unet_model(n_classes=n_classes, IMG_HEIGHT=IMG_HEIGHT, IMG_WIDTH=IMG_WIDTH, IMG_CHANNELS=IMG_CHANNELS)
    # start training
    scores, iou_list, acc_list = {},[],[]
    # initialize model
    model = get_model()
    idx = 0
    X_train,X_val,X_test,y_train,y_val,y_test,y_train_cat,y_val_cat,y_test_cat = create_dataset(idx)
#     model.compile(optimizer='adam', loss=LOSS, metrics=['accuracy'])  

def train_attention(n_classes, IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS ):
    def get_model():
        return attention_unet(n_classes=n_classes, IMG_HEIGHT=IMG_HEIGHT, IMG_WIDTH=IMG_WIDTH, IMG_CHANNELS=IMG_CHANNELS)
    # start training
    scores, iou_list, acc_list = {},[],[]
    # initialize model
    model = get_model()
#     model.compile(optimizer='adam', loss=LOSS, metrics=['accuracy']) 
    
def train_residual(n_classes, IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS):
    def get_model():
        return residual_unet(n_classes=n_classes, IMG_HEIGHT=IMG_HEIGHT, IMG_WIDTH=IMG_WIDTH, IMG_CHANNELS=IMG_CHANNELS)
    # start training
    scores, iou_list, acc_list = {},[],[]
    # initialize model
    model = get_model()
#     model.compile(optimizer='adam', loss=LOSS, metrics=['accuracy']) 


if __name__ == "__main__":
    print(model, dataset, early_stopping)
    n_classes, IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS = 6, 128, 256, 1
    if args.model == "unet":
        train_unet(n_classes, IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS )
    elif args.model == "attention":
        train_attention(n_classes, IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS )
    elif args.model == "residual":
        train_residual(n_classes, IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS )
    