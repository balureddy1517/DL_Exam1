#------------------------------------------------------------------------------------------------------------------
import os
import numpy as np
import random
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf
import sys
from sklearn.metrics import f1_score, cohen_kappa_score, accuracy_score,  matthews_corrcoef
from tensorflow.keras.utils import to_categorical

#------------------------------------------------------------------------------------------------------------------

'''
LAST UPDATE 10/20/2021 LSDR
last update 10/21/2021 lsdr
02/14/2022 am LSDR CHECK CONSISTENCY
02/14/2022 pm LSDR Change result for results

'''
#------------------------------------------------------------------------------------------------------------------

## Process images in parallel
AUTOTUNE = tf.data.AUTOTUNE

## folder "Data" images
## folder "excel" excel file , whatever is there is the file
## get the classes from the excel file
## folder "Documents" readme file

OR_PATH = os.getcwd()
os.chdir("..") # Change to the parent directory
PATH = os.getcwd()
DATA_DIR = os.getcwd() + os.path.sep + 'Data' + os.path.sep
sep = os.path.sep
os.chdir(OR_PATH) # Come back to the folder where the code resides , all files will be left on this directory

n_epoch = 10
BATCH_SIZE = 2
AUGMENT_TIMES = 3

## Image processing
CHANNELS = 3
IMAGE_SIZE = 300

NICKNAME = 'Dean'
#------------------------------------------------------------------------------------------------------------------

def process_target(target_type):
    '''
        1- Multiclass  target = (1...n, text1...textn)
        2- Multilabel target = ( list(Text1, Text2, Text3 ) for each observation, separated by commas )
        3- Binary   target = (1,0)

    :return:
    '''


    class_names = np.sort(xdf_data['target'].unique())

    if target_type == 1:

        x = lambda x: tf.argmax(x == class_names).numpy()

        final_target = xdf_data['target'].apply(x)

        final_target = to_categorical(list(final_target))

        xfinal=[]
        for i in range(len(final_target)):
            joined_string = ",".join(str(int(e)) for e in  (final_target[i]))
            xfinal.append(joined_string)
        final_target = xfinal

        xdf_data['target_class'] = final_target


    if target_type == 2:
        target = np.array(xdf_data['target'].apply(lambda x: x.split(",")))

        xdepth = len(class_names)

        final_target = tf.one_hot(target, xdepth)

        xfinal = []
        if len(final_target) ==0:
            xerror = 'Could not process Multilabel'
        else:
            for i in range(len(final_target)):
                joined_string = ",".join( str(e) for e in final_target[i])
                xfinal.append(joined_string)
            final_target = xfinal

        xdf_data['target_class'] = final_target

    if target_type == 3:
        # target_class is already done
        pass

    return class_names
#------------------------------------------------------------------------------------------------------------------


def augment_image(image):
    # Example augmentations: flip, adjust brightness, rotation
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_brightness(image, max_delta=0.2)
    image = tf.image.random_contrast(image, lower=0.8, upper=1.2)
    image = image / 255.0  # Rescale the image to be between 0 and 1

    # Ensure the pixel values are clipped between 0 and 1 in case the augmentations push them out of bounds
    image = tf.clip_by_value(image, 0.0, 1.0)
    return image
def process_path(feature, target):
    '''
          feature is the path and id of the image
          target is the result
          returns the image and the target as label
    '''

    label = target

    file_path = feature
    print(feature)
    print(label.shape)
    img = tf.io.read_file(file_path)



    img = tf.io.decode_image(img, channels=CHANNELS, expand_animations=False)


    img = tf.image.resize( img, [IMAGE_SIZE, IMAGE_SIZE])


    # augmentation


    # img = tf.reshape(img, [-1])


    return img, label


def augment_if_needed(image_path, label):
    # Check the argmax of the label
    label_argmax = tf.argmax(label)

    # Use tf.reduce_any with tf.equal to check if label_argmax is in [3, 4, 5]
    augment_condition = tf.reduce_any(tf.equal(label_argmax, [3, 4, 5]))

    def augment_images():
        img, lbl = process_path(image_path, label)  # Image in 3D shape
        augmented_images = [(img, lbl)]

        for _ in range(AUGMENT_TIMES):
            aug_img = augment_image(img)  # Augment in 3D shape
            augmented_images.append((aug_img, lbl))

        # Reshape after augmentations if needed
        images = tf.stack([tf.reshape(x[0], [-1]) for x in augmented_images])  # Flattened image
        labels = tf.stack([x[1] for x in augmented_images])
        return tf.data.Dataset.from_tensor_slices((images, labels))

    def return_original():
        img, lbl = process_path(image_path, label)
        img = tf.reshape(img, [-1])  # Flatten the image for model input
        return tf.data.Dataset.from_tensors((img, lbl))

    return tf.cond(augment_condition, augment_images, return_original)


#------------------------------------------------------------------------------------------------------------------

def get_target(num_classes):
    '''
    Get the target from the dataset
    1 = multiclass
    2 = multilabel
    3 = binary
    '''

    y_target = np.array(xdf_dset['target_class'].apply(lambda x: ([int(i) for i in str(x).split(",")])))

    end = np.zeros(num_classes)
    for s1 in y_target:
        end = np.vstack([end, s1])

    y_target = np.array(end[1:])


    return y_target
#------------------------------------------------------------------------------------------------------------------
def read_data(num_classes):
    '''
          reads the dataset and process the target
    '''

    ds_inputs = np.array(DATA_DIR + xdf_dset['id'])
    ds_targets = get_target(num_classes)


    list_ds = tf.data.Dataset.from_tensor_slices((ds_inputs,ds_targets)) # creates a tensor from the image paths and targets



    # final_ds = list_ds.map(process_path, num_parallel_calls=AUTOTUNE).batch(BATCH_SIZE)
    augmented_dataset = list_ds.flat_map(lambda image_path, label: augment_if_needed(image_path, label))
    final_ds = augmented_dataset.batch(BATCH_SIZE)

    return final_ds

def test_read_data(num_classes):
    '''
          reads the dataset and process the target
    '''

    ds_inputs = np.array(DATA_DIR + xdf_dset['id'])
    ds_targets = get_target(num_classes)

    list_ds = tf.data.Dataset.from_tensor_slices((ds_inputs,ds_targets)) # creates a tensor from the image paths and targets

    final_ds = list_ds.map(test_process_path, num_parallel_calls=AUTOTUNE).batch(BATCH_SIZE)

    return final_ds

def test_process_path(feature, target):
    '''
          feature is the path and id of the image
          target is the result
          returns the image and the target as label
    '''

    label = target

    file_path = feature
    print(feature)
    print(label.shape)
    img = tf.io.read_file(file_path)



    img = tf.io.decode_image(img, channels=CHANNELS, expand_animations=False)


    img = tf.image.resize( img, [IMAGE_SIZE, IMAGE_SIZE])


    # augmentation


    img = tf.reshape(img, [-1])


    return img, label
#------------------------------------------------------------------------------------------------------------------

def save_model(model):
    '''
         receives the model and print the summary into a .txt file
    '''
    with open('summary_{}.txt'.format(NICKNAME), 'w') as fh:
        # Pass the file handle in as a lambda function to make it callable
        model.summary(print_fn=lambda x: fh.write(x + '\n'))
#------------------------------------------------------------------------------------------------------------------

def model_definition():
    # Define a Keras sequential model
    model = tf.keras.Sequential()

    # First dense layer with more neurons and batch normalization
    model.add(tf.keras.layers.Dense(512, activation='relu', input_shape=(INPUTS_r,)))
    model.add(tf.keras.layers.BatchNormalization())  # Batch Normalization
    model.add(tf.keras.layers.Dropout(0.5))  # Dropout layer with 50% rate

    # Additional layers with increased neurons and regularization
    model.add(tf.keras.layers.Dense(256, activation='relu'))
    model.add(tf.keras.layers.BatchNormalization())  # Batch Normalization
    model.add(tf.keras.layers.Dropout(0.5))  # Dropout layer with 50% rate

    model.add(tf.keras.layers.Dense(128, activation='relu'))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Dropout(0.5))

    model.add(tf.keras.layers.Dense(64, activation='relu'))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Dropout(0.5))

    # Final layer, OUTPUTS_a is the number of targets (softmax for classification)
    model.add(tf.keras.layers.Dense(OUTPUTS_a, activation='softmax'))

    # Use SGD with momentum
    optimizer = tf.keras.optimizers.SGD(learning_rate=0.001, momentum=0.9)

    # Compile the model with the SGD optimizer and categorical crossentropy loss
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])


    save_model(model)  # Save model summary to a file
    return model
#------------------------------------------------------------------------------------------------------------------

def train_func(train_ds):
    '''
        train the model
    '''

    #early_stop = tf.keras.callbacks.EarlyStopping(monitor='accuracy', patience =100)
    check_point = tf.keras.callbacks.ModelCheckpoint('model_{}.keras'.format(NICKNAME), monitor='accuracy', save_best_only=True)
    final_model = model_definition()

    #final_model.fit(train_ds,  epochs=n_epoch, callbacks=[early_stop, check_point])
    # final_model.fit(train_ds,  epochs=n_epoch, callbacks=[check_point])
    final_model.fit(train_ds, epochs=n_epoch, callbacks=[check_point], batch_size=32)


#------------------------------------------------------------------------------------------------------------------

def predict_func(test_ds):
    '''
        predict fumction
    '''

    final_model = tf.keras.models.load_model('model_{}.keras'.format(NICKNAME))
    res = final_model.predict(test_ds)
    xres = [ tf.argmax(f).numpy() for f in res]
    xdf_dset['results'] = xres
    xdf_dset.to_excel('results_{}.xlsx'.format(NICKNAME), index=False)
#------------------------------------------------------------------------------------------------------------------

def metrics_func(metrics, aggregates=[]):
    '''
    multiple functiosn of metrics to call each function
    f1, cohen, accuracy, mattews correlation
    list of metrics: f1_micro, f1_macro, f1_avg, coh, acc, mat
    list of aggregates : avg, sum
    :return:
    '''

    def f1_score_metric(y_true, y_pred, type):
        '''
            type = micro,macro,weighted,samples
        :param y_true:
        :param y_pred:
        :param average:
        :return: res
        '''
        res = f1_score(y_true, y_pred, average=type)
        print("f1_score {}".format(type), res)
        return res

    def cohen_kappa_metric(y_true, y_pred):
        res = cohen_kappa_score(y_true, y_pred)
        print("cohen_kappa_score", res)
        return res

    def accuracy_metric(y_true, y_pred):
        res = accuracy_score(y_true, y_pred)
        print("accuracy_score", res)
        return res

    def matthews_metric(y_true, y_pred):
        res = matthews_corrcoef(y_true, y_pred)
        print('mattews_coef', res)
        return res


    # For multiclass

    x = lambda x: tf.argmax(x == class_names).numpy()
    y_true = np.array(xdf_dset['target'].apply(x))
    y_pred = np.array(xdf_dset['results'])

    # End of Multiclass

    xcont = 1
    xsum = 0
    xavg = 0

    for xm in metrics:
        if xm == 'f1_micro':
            # f1 score average = micro
            xmet = f1_score_metric(y_true, y_pred, 'micro')
        elif xm == 'f1_macro':
            # f1 score average = macro
            xmet = f1_score_metric(y_true, y_pred, 'macro')
        elif xm == 'f1_weighted':
            # f1 score average =
            xmet = f1_score_metric(y_true, y_pred, 'weighted')
        elif xm == 'coh':
             # Cohen kappa
            xmet = cohen_kappa_metric(y_true, y_pred)
        elif xm == 'acc':
            # Accuracy
            xmet =accuracy_metric(y_true, y_pred)
        elif xm == 'mat':
            # Matthews
            xmet =matthews_metric(y_true, y_pred)
        else:
            xmet =print('Metric does not exist')

        xsum = xsum + xmet
        xcont = xcont + 1

    if 'sum' in aggregates:
        print('Sum of Metrics : ', xsum )
    if 'avg' in aggregates and xcont > 0:
        print('Average of Metrics : ', xsum/xcont)
    # Ask for arguments for each metric
#------------------------------------------------------------------------------------------------------------------

def main():
    global xdf_data, class_names, INPUTS_r, OUTPUTS_a, xdf_dset

    for file in os.listdir(PATH+os.path.sep + "excel"):
        if file[-5:] == '.xlsx':
            FILE_NAME = PATH + os.path.sep + "excel" + os.path.sep + file

    # Reading and filtering Excel file
    xdf_data = pd.read_excel(FILE_NAME)

    class_names= process_target(1)  # 1: Multiclass 2: Multilabel 3:Binary

    INPUTS_r = IMAGE_SIZE * IMAGE_SIZE * CHANNELS
    OUTPUTS_a = len(class_names)

    ## Processing Train dataset

    xdf_dset = xdf_data[xdf_data["split"] == 'train'].copy()
    print(len(xdf_dset))
    class_counts = xdf_dset['target'].value_counts()
    print(class_counts)
    #
    train_ds = read_data( OUTPUTS_a )

    train_func(train_ds)

    # Preprocessing Test dataset

    xdf_dset = xdf_data[xdf_data["split"] == 'test'].copy()

    test_ds= test_read_data(OUTPUTS_a)
    predict_func(test_ds)

    ## Metrics Function over the result of the test dataset
    list_of_metrics = ['f1_macro', 'coh','acc']
    list_of_agg = ['avg']
    metrics_func(list_of_metrics, list_of_agg)

# ------------------------------------------------------------------------------------------------------------------


if __name__ == "__main__":

    main()
#------------------------------------------------------------------------------------------------------------------

