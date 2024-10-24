import numpy as np
import keras
from tensorflow.keras import layers
import os
import pandas as pd
import tensorflow as tf
from sklearn.metrics import f1_score, cohen_kappa_score, accuracy_score, matthews_corrcoef
from tensorflow.keras.utils import to_categorical
from sklearn.utils.class_weight import compute_class_weight
from sklearn.preprocessing import LabelEncoder
from openpyxl import Workbook

# Hyperparameters
weight_decay = 0.0001
batch_size = 64
num_epochs = 50
dropout_rate = 0.2
image_size = 64
patch_size = 8
num_patches = (image_size // patch_size) ** 2
embedding_dim = 256
num_blocks = 4
num_classes = 5
input_shape = (64, 64, 3)
learning_rate = 0.001
model_name='Dean'



# Data augmentation
data_augmentation = keras.Sequential(
    [
        layers.Normalization(),
        layers.Resizing(image_size, image_size),
        layers.RandomFlip("horizontal"),
        layers.RandomZoom(height_factor=0.2, width_factor=0.2),
    ],
    name="data_augmentation",
)


# Function to process the target
def process_target(target_type):
    class_names = np.sort(xdf_data['target'].unique())
    if target_type == 1:
        # Use LabelEncoder to convert class names to integers, then one-hot encode
        le = LabelEncoder()
        xdf_data['target_class'] = le.fit_transform(xdf_data['target'])
        final_target = to_categorical(xdf_data['target_class'], num_classes=num_classes)
        xdf_data['target_class'] = list(final_target)
    return class_names


# Process image paths and corresponding labels
def process_path(feature, target):
    img = tf.io.read_file(feature)
    img = tf.io.decode_image(img, channels=3, expand_animations=False)
    img = tf.image.resize(img, [image_size, image_size])
    img = img / 255.0  # Normalize to [0, 1]

    label = target
    return img, label


# Get the target values for the dataset
def get_target():
    y_target = np.array(xdf_dset['target_class'].apply(lambda x: [int(i) for i in x]))
    return np.array(y_target)


# Read data from the dataset
def read_data():
    ds_inputs = np.array(DATA_DIR + xdf_dset['id'])
    print("unique split",xdf_dset['split'].unique())
    ds_targets = get_target()

    images = []
    labels = []

    for img_path, label in zip(ds_inputs, ds_targets):
        img, lbl = process_path(img_path, label)
        images.append(img)
        labels.append(lbl)

    return np.array(images), np.array(labels)


# Metrics function for evaluation
def metrics_func(metrics, aggregates=[]):
    '''
    multiple functions of metrics to call each function
    f1, cohen, accuracy, mattews correlation
    list of metrics: f1_micro, f1_macro, f1_avg, coh, acc, mat
    list of aggregates : avg, sum
    :return:
    '''

    def f1_score_metric(y_true, y_pred, type):
        res = f1_score(y_true, y_pred, average=type)
        print(f"f1_score ({type}):", res)
        return res

    def cohen_kappa_metric(y_true, y_pred):
        res = cohen_kappa_score(y_true, y_pred)
        print("cohen_kappa_score:", res)
        return res

    def accuracy_metric(y_true, y_pred):
        res = accuracy_score(y_true, y_pred)
        print("accuracy_score:", res)
        return res

    def matthews_metric(y_true, y_pred):
        res = matthews_corrcoef(y_true, y_pred)
        print('matthews_corrcoef:', res)
        return res

    # Prepare y_true and y_pred
    x = lambda x: np.argmax(x == class_names)
    y_true = np.array(xdf_dset['target'].apply(x))
    y_pred = np.array(xdf_dset['results'])

    # Aggregate metrics
    xsum = 0
    xcont = 1

    for xm in metrics:
        if xm == 'f1_micro':
            xmet = f1_score_metric(y_true, y_pred, 'micro')
        elif xm == 'f1_macro':
            xmet = f1_score_metric(y_true, y_pred, 'macro')
        elif xm == 'f1_weighted':
            xmet = f1_score_metric(y_true, y_pred, 'weighted')
        elif xm == 'coh':
            xmet = cohen_kappa_metric(y_true, y_pred)
        elif xm == 'acc':
            xmet = accuracy_metric(y_true, y_pred)
        elif xm == 'mat':
            xmet = matthews_metric(y_true, y_pred)
        else:
            print('Metric does not exist')
            continue

        xsum += xmet
        xcont += 1

    if 'sum' in aggregates:
        print('Sum of Metrics:', xsum)
    if 'avg' in aggregates and xcont > 0:
        print('Average of Metrics:', xsum / xcont)


# Build the classifier
def build_classifier(blocks, positional_encoding=False):
    inputs = layers.Input(shape=input_shape)
    augmented = data_augmentation(inputs)
    patches = Patches(patch_size)(augmented)
    x = layers.Dense(units=embedding_dim)(patches)

    if positional_encoding:
        x = x + PositionEmbedding(sequence_length=num_patches)(x)

    x = blocks(x)
    representation = layers.GlobalAveragePooling1D()(x)
    representation = layers.Dropout(rate=dropout_rate)(representation)
    logits = layers.Dense(num_classes)(representation)

    return keras.Model(inputs=inputs, outputs=logits)


# Function to run experiment and call metrics function
def run_experiment(model, model_name):
    optimizer = keras.optimizers.AdamW(
        learning_rate=learning_rate,
        weight_decay=weight_decay,
    )
    model.compile(
        optimizer=optimizer,
        loss=keras.losses.CategoricalCrossentropy(from_logits=True),
        metrics=[
            keras.metrics.CategoricalAccuracy(name="acc"),
            keras.metrics.TopKCategoricalAccuracy(5, name="top5-acc"),
        ],
    )
    reduce_lr = keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss", factor=0.5, patience=5
    )
    early_stopping = keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=10, restore_best_weights=True
    )

    history = model.fit(
        x=x_train,
        y=y_train,
        batch_size=batch_size,
        epochs=num_epochs,
        validation_split=0.1,
        callbacks=[early_stopping, reduce_lr],
        verbose=0,
    )

    # Make predictions
    y_pred_prob = model.predict(x_test)
    y_pred = np.argmax(y_pred_prob, axis=1)

    # Store the predictions in the dataframe
    xdf_dset['results'] = y_pred

    xdf_dset.to_excel('results_{}.xlsx'.format(model_name), index=False)

    # Call the metrics function with desired metrics and aggregates
    list_of_metrics = ['f1_macro', 'coh', 'acc']
    list_of_agg = ['avg']
    metrics_func(list_of_metrics, list_of_agg)

    # Save the model with dynamic name
    model_filename = f'model_{model_name}.keras'
    model.save(model_filename)
    print(f"Model saved as: {model_filename}")

    # Save the model summary to a dynamically named text file
    summary_filename = f'summary_{model_name}.txt'
    with open(summary_filename, 'w') as f:
        model.summary(print_fn=lambda x: f.write(x + '\n'))
    print(f"Model summary saved as: {summary_filename}")

    results_filename = f'results_{model_name}.xlsx'

    # Evaluate the model
    results = model.evaluate(x_test, y_test, return_dict=True)

    # Create a DataFrame for evaluation metrics
    metrics_df = pd.DataFrame(list(results.items()), columns=['Metric', 'Value'])

    # Create a DataFrame for image details and predictions
    image_details_df = pd.DataFrame({
        'Image_ID': xdf_dset['id'],  # Assuming 'id' column contains image file paths or unique IDs
        'True_Label': y_test.argmax(axis=1),  # Assuming y_test is one-hot encoded
        'Predicted_Label': y_pred
    })

    # Create an Excel writer object
    with pd.ExcelWriter(results_filename, engine='openpyxl') as writer:
        # Write metrics to the first sheet
        metrics_df.to_excel(writer, sheet_name='Metrics', index=False)

        # Write image details and predictions to the second sheet
        image_details_df.to_excel(writer, sheet_name='Image_Details_Predictions', index=False)

    print(f"Results saved as: {results_filename}")

    return history

# Patches Layer
@keras.saving.register_keras_serializable()
class Patches(layers.Layer):
    def __init__(self, patch_size, **kwargs):
        super().__init__(**kwargs)
        self.patch_size = patch_size

    def call(self, x):
        patches = tf.image.extract_patches(images=x,
                                           sizes=[1, self.patch_size, self.patch_size, 1],
                                           strides=[1, self.patch_size, self.patch_size, 1],
                                           rates=[1, 1, 1, 1],
                                           padding='VALID')
        batch_size = tf.shape(patches)[0]
        num_patches = tf.shape(patches)[1] * tf.shape(patches)[2]
        patch_dim = tf.shape(patches)[3]
        return tf.reshape(patches, (batch_size, num_patches, patch_dim))


# PositionEmbedding Layer
class PositionEmbedding(keras.layers.Layer):
    def __init__(
            self,
            sequence_length,
            initializer="glorot_uniform",
            **kwargs,
    ):
        super().__init__(**kwargs)
        if sequence_length is None:
            raise ValueError("`sequence_length` must be an Integer, received `None`.")
        self.sequence_length = int(sequence_length)
        self.initializer = keras.initializers.get(initializer)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "sequence_length": self.sequence_length,
                "initializer": keras.initializers.serialize(self.initializer),
            }
        )
        return config

    def build(self, input_shape):
        feature_size = input_shape[-1]
        self.position_embeddings = self.add_weight(
            name="embeddings",
            shape=[self.sequence_length, feature_size],
            initializer=self.initializer,
            trainable=True,
        )

        super().build(input_shape)

    def call(self, inputs, start_index=0):
        shape = tf.shape(inputs)
        feature_length = shape[-1]
        sequence_length = shape[-2]
        position_embeddings = tf.convert_to_tensor(self.position_embeddings)
        position_embeddings = tf.slice(
            position_embeddings,
            (start_index, 0),
            (sequence_length, feature_length),
        )
        return tf.broadcast_to(position_embeddings, shape)


# MLPMixerLayer Class
@keras.saving.register_keras_serializable()
class MLPMixerLayer(layers.Layer):
    def __init__(self, num_patches, hidden_units, dropout_rate, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.mlp1 = keras.Sequential(
            [
                layers.BatchNormalization(),
                layers.Dense(units=num_patches, activation="gelu"),
                layers.Dense(units=num_patches),
                layers.Dropout(rate=dropout_rate),
            ]
        )
        self.mlp2 = keras.Sequential(
            [
                layers.BatchNormalization(),
                layers.Dense(units=num_patches, activation="gelu"),
                layers.Dense(units=hidden_units),
                layers.Dropout(rate=dropout_rate),
            ]
        )
        self.normalize = layers.LayerNormalization(epsilon=1e-6)

    def build(self, input_shape):
        return super().build(input_shape)

    def call(self, inputs):
        x = self.normalize(inputs)
        x_channels = tf.transpose(x, perm=(0, 2, 1))
        mlp1_outputs = self.mlp1(x_channels)
        mlp1_outputs = tf.transpose(mlp1_outputs, perm=(0, 2, 1))
        x = mlp1_outputs + inputs
        x_patches = self.normalize(x)
        mlp2_outputs = self.mlp2(x_patches)
        x = x + mlp2_outputs
        return x


# Set paths and directories
OR_PATH = os.getcwd()
os.chdir("..")
PATH = os.getcwd()
DATA_DIR = os.path.join(PATH, 'Data') + os.path.sep
os.chdir(OR_PATH)

# Load data from excel
for file in os.listdir(PATH + os.path.sep + "excel"):
    if file.endswith('.xlsx'):
        FILE_NAME = os.path.join(PATH, "excel", file)
xdf_data = pd.read_excel(FILE_NAME)

# Process target and training data
class_names = process_target(1)
OUTPUTS_a = len(class_names)

xdf_dset = xdf_data[xdf_data["split"] == 'train'].copy()
x_train, y_train = read_data()
print(x_train.shape, y_train.shape)

xdf_dset = xdf_data[xdf_data["split"] == 'test'].copy()
x_test, y_test = read_data()
print(x_test.shape, y_test.shape)

# Build MLP Mixer model
mlpmixer_blocks = keras.Sequential(
    [MLPMixerLayer(num_patches, embedding_dim, dropout_rate) for _ in range(num_blocks)]
)
mlpmixer_classifier = build_classifier(mlpmixer_blocks)


# Run the experiment
history = run_experiment(mlpmixer_classifier,model_name)
