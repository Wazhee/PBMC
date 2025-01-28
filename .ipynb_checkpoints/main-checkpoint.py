import argparse
from models import multi_unet_model, attention_unet, residual_unet
from tqdm import tqdm
import tensorflow as tf
from evaluate import *
from dataset import *
from clean_results import *
from tensorflow.keras.optimizers import Adam

parser = argparse.ArgumentParser()
parser.add_argument('-train', action='store_true')
parser.add_argument('-model', default='unet', choices=['unet', 'attention', 'residual'])
parser.add_argument('-dataset', default='fib', choices=['fib'])
parser.add_argument('-augment', action='store_true') # enable data augmentation
parser.add_argument('-early_stopping', action='store_true') # enable early_stopping
parser.add_argument('-gpu', type=int, default=0, choices=[0, 1, 2, 3])
parser.add_argument('-epochs', type=int, default=100)
parser.add_argument('-loss', default='categorical_crossentropy', choices=['dice_crossentropy', 'mae', 'mse', 'categorical_crossentropy', 'dice', 'focal'])
parser.add_argument('-loss2', default=None, choices=['mae', 'mse', 'categorical_crossentropy', 'dice', 'focal'])
parser.add_argument('-save_dir', default='models', choices=['unet', 'attention', 'residual'])
parser.add_argument('-test', action='store_true')
parser.add_argument('-custom', action='store_true')
parser.add_argument('-img_path', type=str, default="../datasets/100nm/Images/Aligned SEM Images/")
parser.add_argument('-msk_path', type=str, default="../datasets/100nm/Images/fake_labels/")

args = parser.parse_args()
dataset = args.dataset
early_stopping = args.early_stopping
model = args.model

gpus = tf.config.list_physical_devices('GPU')
if len(gpus) > 0: 
    tf.config.experimental.set_visible_devices(gpus[args.gpu], 'GPU')
    
# Define the Dice score metric
def dice_score(y_true, y_pred):
    y_true_f = tf.keras.backend.flatten(y_true)
    y_pred_f = tf.keras.backend.flatten(y_pred)
    intersection = tf.keras.backend.sum(y_true_f * y_pred_f)
    return (2. * intersection + 1e-6) / (tf.keras.backend.sum(y_true_f) + tf.keras.backend.sum(y_pred_f) + 1e-6)

def get_loss(loss, secondary_loss = None):
    specialty = {'focal': focal_loss, 'dice': dice_coef_loss}
    if loss == 'mse' or loss == 'categorical_crossentropy' or loss == 'mae':
        return loss
    elif loss == 'dice':
        return dice_coef_loss
    elif loss == 'focal':
        return focal_loss
    elif loss == 'dice_crossentropy':
        return combined_loss
        
def train_unet():
    save_dir = f"EPOCHS{args.epochs}_{args.model.upper()}_{args.loss}_es{args.early_stopping}_aug{args.augment}/"
    model_savepath,model_checkpoint = f"../results/models/{save_dir}/", "../results/checkpoints"
    BATCHSIZE, EPOCHS = 16, args.epochs
    if args.loss2 is not None:
        LOSS = [get_loss(args.loss), get_loss(args.loss2)]
    else:
        LOSS = get_loss(args.loss) # if 
    
    if args.test:
        print("Test script enabled...\n")
        save_dir = "Testing_Unet"
        model_path = "../results/models/EPOCHS300_UNET_focal_esFalse_augTrue/unet.hdf5"
        for idx in tqdm(range(150)):
            X_train,X_val,X_test,y_train,y_val,y_test,y_train_cat,y_val_cat,y_test_cat = create_custom_dataset(img=args.img_path, msk=args.msk_path, number=idx, augment=args.augment)
            print(f"X_train.shape: {X_train.shape}, X_val.shape: {X_val.shape}, X_test.shape: {X_test.shape}")
            IMG_HEIGHT,IMG_WIDTH,IMG_CHANNELS = X_train.shape[1],X_train.shape[2],X_train.shape[3]
            def get_model():
                return multi_unet_model(n_classes=n_classes, IMG_HEIGHT=IMG_HEIGHT, IMG_WIDTH=IMG_WIDTH, IMG_CHANNELS=IMG_CHANNELS)
            # start training
            learning_rate = 0.0001  # Specify your desired learning rate
            model = get_model()        # initialize model
            model.load_weights(model_path)
            scores = performance_evaluation(model, X_test, y_test, n_classes, {}, idx, save_dir) # calculate results
    else:
        for idx in tqdm(range(65)):
            X_train,X_val,X_test,y_train,y_val,y_test,y_train_cat,y_val_cat,y_test_cat = create_dataset(idx, augment=args.augment)
            print(f"X_train.shape: {X_train.shape}, X_val.shape: {X_val.shape}, X_test.shape: {X_test.shape}")
            IMG_HEIGHT,IMG_WIDTH,IMG_CHANNELS = X_train.shape[1],X_train.shape[2],X_train.shape[3]
            def get_model():
                return multi_unet_model(n_classes=n_classes, IMG_HEIGHT=IMG_HEIGHT, IMG_WIDTH=IMG_WIDTH, IMG_CHANNELS=IMG_CHANNELS)
            # start training
            scores, iou_list, acc_list = {},[],[]
            # initialize model
            learning_rate = 0.0001  # Specify your desired learning rate
            model = get_model()
            model.compile(optimizer=Adam(learning_rate=learning_rate), loss=LOSS, metrics=['accuracy', dice_score])
    #         model = get_model()
    #         model.compile(optimizer='adam', loss=LOSS, metrics=['accuracy'])  

            early_stopping = EarlyStopping(monitor='val_loss', patience=30, restore_best_weights=True) # implement early_stopping mechanism
            history = model.fit(X_train, y_train_cat, 
                                batch_size = BATCHSIZE, 
                                verbose=2, 
                                epochs=EPOCHS, 
                                validation_data=(X_val, y_val_cat), 
                                callbacks=[early_stopping],
                                #class_weight=class_weights,
                                shuffle=True)

            scores = performance_evaluation(model, X_test, y_test, n_classes, scores,idx,save_dir) # calculate results
            # save results of K-fold validation
            create_csv(scores, idx, save_dir)
            # save model
            model.save(f"{model_savepath}{args.model}.hdf5")
            print("Model successfully trained and saved!")
        


def train_attention():
    save_dir = f"EPOCHS{args.epochs}_{args.model.upper()}_{args.loss}_es{args.early_stopping}_aug{args.augment}/"
    model_savepath,model_checkpoint = f"../results/models/{save_dir}/", "../results/checkpoints"
    BATCHSIZE, EPOCHS = 16, args.epochs
    if args.loss2 is not None:
        LOSS = [get_loss(args.loss), get_loss(args.loss2)]
    else:
        LOSS = get_loss(args.loss) # if 
    
    for idx in tqdm(range(65)):
        X_train,X_val,X_test,y_train,y_val,y_test,y_train_cat,y_val_cat,y_test_cat = create_dataset(idx, augment=args.augment)
        print(f"X_train.shape: {X_train.shape}, X_val.shape: {X_val.shape}, X_test.shape: {X_test.shape}")
        IMG_HEIGHT,IMG_WIDTH,IMG_CHANNELS = X_train.shape[1],X_train.shape[2],X_train.shape[3]
        def get_model():
            return attention_unet(n_classes=n_classes, IMG_HEIGHT=IMG_HEIGHT, IMG_WIDTH=IMG_WIDTH, IMG_CHANNELS=IMG_CHANNELS)
        # start training
        scores, iou_list, acc_list = {},[],[]
        # initialize model
        learning_rate = 0.0001  # Specify your desired learning rate
        model = get_model()
        model.compile(optimizer=Adam(learning_rate=learning_rate), loss=LOSS, metrics=['accuracy', dice_score])

        early_stopping = EarlyStopping(monitor='val_loss', patience=30, restore_best_weights=True) # implement early_stopping mechanism
        history = model.fit(X_train, y_train_cat, 
                            batch_size = BATCHSIZE, 
                            verbose=2, 
                            epochs=EPOCHS, 
                            validation_data=(X_val, y_val_cat), 
                            callbacks=[early_stopping],
                            #class_weight=class_weights,
                            shuffle=True)

        scores = performance_evaluation(model, X_test, y_test, n_classes, scores,idx,save_dir) # calculate results
        # save results of K-fold validation
        create_csv(scores, idx, save_dir)
        # save model
        model.save(f"{model_savepath}{args.model}.hdf5")
        print("Model successfully trained and saved!")
    
def train_residual():
    save_dir = f"EPOCHS{args.epochs}_{args.model.upper()}_{args.loss}_es{args.early_stopping}_aug{args.augment}/"
    model_savepath,model_checkpoint = f"../results/models/{save_dir}/", "../results/checkpoints"
    BATCHSIZE, EPOCHS = 16, args.epochs
    if args.loss2 is not None:
        LOSS = [get_loss(args.loss), get_loss(args.loss2)]
    else:
        LOSS = get_loss(args.loss) # if 
    
    for idx in tqdm(range(65)):
        X_train,X_val,X_test,y_train,y_val,y_test,y_train_cat,y_val_cat,y_test_cat = create_dataset(idx, augment=args.augment)
        print(f"X_train.shape: {X_train.shape}, X_val.shape: {X_val.shape}, X_test.shape: {X_test.shape}")
        IMG_HEIGHT,IMG_WIDTH,IMG_CHANNELS = X_train.shape[1],X_train.shape[2],X_train.shape[3]
        def get_model():
            return residual_unet(n_classes=n_classes, IMG_HEIGHT=IMG_HEIGHT, IMG_WIDTH=IMG_WIDTH, IMG_CHANNELS=IMG_CHANNELS)
        # start training
        scores, iou_list, acc_list = {},[],[]
        # initialize model
        learning_rate = 0.0001  # Specify your desired learning rate
        model = get_model()
        model.compile(optimizer=Adam(learning_rate=learning_rate), loss=LOSS, metrics=['accuracy', dice_score])

        early_stopping = EarlyStopping(monitor='val_loss', patience=30, restore_best_weights=True) # implement early_stopping mechanism
        history = model.fit(X_train, y_train_cat, 
                            batch_size = BATCHSIZE, 
                            verbose=2, 
                            epochs=EPOCHS, 
                            validation_data=(X_val, y_val_cat), 
                            callbacks=[early_stopping],
                            #class_weight=class_weights,
                            shuffle=True)

        scores = performance_evaluation(model, X_test, y_test, n_classes, scores,idx,save_dir) # calculate results
        # save results of K-fold validation
        create_csv(scores, idx, save_dir)
        # save model
        model.save(f"{model_savepath}{args.model}.hdf5")
        print("Model successfully trained and saved!") 


if __name__ == "__main__":
    print(f"\n\nTraining {model.upper()} model on {dataset.upper()} dataset, w/ early stopping set to {early_stopping}...\n")
    if args.model == "unet":
        train_unet()
    elif args.model == "attention":
        train_attention()
    elif args.model == "residual":
        train_residual()