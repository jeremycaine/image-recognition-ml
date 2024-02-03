import os
import sys

import tensorflow as tf

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import ibm_boto3
from ibm_botocore.client import Config
from ibm_botocore.exceptions import ClientError

import tempfile
import pathlib

# credentials
COS_API_KEY_ID = os.getenv('COS_API_KEY_ID')
COS_ENDPOINT = os.getenv('COS_ENDPOINT', 'https://s3.eu-gb.cloud-object-storage.appdomain.cloud')
COS_AUTH_ENDPOINT = os.getenv('COS_AUTH_ENDPOINT', 'https://iam.cloud.ibm.com/identity/token')
COS_SERVICE_CRN = os.getenv('COS_SERVICE_CRN', 'crn:v1:bluemix:public:cloud-object-storage:global:a/b71ac2564ef0b98f1032d189795994dc:875e3790-53c1-40b0-9943-33b010521174::')
COS_STORAGE_CLASS = os.getenv('COS_STORAGE_CLASS','eu-gb-smart')

# file vars
bucket_name = os.getenv('BUCKET_NAME', 'mnist-model')
h5_file_name = os.getenv('H5_FILE_NAME', 'mnist-model.h5')
train_csv = os.getenv('TRAIN_CSV', 'mnist_train.csv')
test_csv = os.getenv('TEST_CSV', 'mnist_test.csv')

def log(e):
    print("{0}\n".format(e))

def get_file(file_name): 
    try:
        # Create a temporary file
        with tempfile.NamedTemporaryFile() as f:
            # Get the pathlib.Path object
            print("download from COS into temp file: ", f.name)
            path = pathlib.Path(f.name)

            # and file name
            fn = path.name

            cos_cli.download_file(bucket_name, file_name, fn) 
            return fn
    except ClientError as be:
        log(be)
        sys.exit(1)
    except Exception as e:
        log("Unable to get file: {0}".format(e))
        sys.exit(1)

def save_file(path_object, file_name): 
    try:
        with open(path_object, 'rb') as file:
            # Use the file-like object here
            # For example, you can read its contents
            contents = file.read()            
            cos_cli.put_object(
                Bucket=bucket_name,
                Key=file_name,
                Body=contents) 

        return 0
    except ClientError as be:
        log(be)
        sys.exit(1)
    except Exception as e:
        log("Unable to put file: {0}".format(e))
        sys.exit(1)

log("start")

# create cloud objec storage connection
cos_cli = ibm_boto3.client("s3",
    ibm_api_key_id=COS_API_KEY_ID,
    ibm_service_instance_id=COS_SERVICE_CRN,
    ibm_auth_endpoint=COS_AUTH_ENDPOINT,
    config=Config(signature_version="oauth"),
    endpoint_url=COS_ENDPOINT
)

test_file = get_file(test_csv)
print(test_file)
test_df: pd.DataFrame = pd.read_csv(test_file, header=None)
test_features: np.ndarray = test_df.loc[:, 1:].values

test_features = test_features.reshape((test_features.shape[0], 28, 28, 1))
test_features = test_features / 255.0
test_labels: np.ndarray = test_df[0].values

# -- Build the model
train_file = get_file(test_csv)
print(train_file)
train_df: pd.DataFrame = pd.read_csv(train_file, header=None)
#test_df: pd.DataFrame = pd.read_csv(get_file(test_csv), header=None)
log("data loaded")

# split data set, taking off the labels e.g. '7' (first element), from the others 
train_features: np.ndarray = train_df.loc[:, 1:].values

# shape the dataset into 60,000 28x28x1
# 60k data items, 28x28 pixels, could be 1 to 3 element, but we only need 1 element (grayscale)
train_features = train_features.reshape((train_features.shape[0], 28, 28, 1))

# each data item has a value between 0 and 255, so to normalise and make value betwee 0 and 1
train_features = train_features / 255.0

# get the labels
train_labels: np.ndarray = train_df[0].values

# manipulate and normailse test data set and get the labels
test_features: np.ndarray = test_df.loc[:, 1:].values
test_features = test_features.reshape((test_features.shape[0], 28, 28, 1))
test_features = test_features / 255.0
test_labels: np.ndarray = test_df[0].values

# create the model
# Sequential - outputs are sent to input of next layer - aka forward progression
model: tf.keras.models.Sequential = tf.keras.models.Sequential()

# add layer performing Convolution over a 2d grid
model.add(tf.keras.layers.Conv2D(filters=32, kernel_size=(3, 3), input_shape=(28, 28, 1)))

# add layer pooling the max value in the 2x2 grid
model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=2))

# add layer to flatten 2d grid into a single array
model.add(tf.keras.layers.Flatten())

# add a dense layer which classifies the data using an activation function
model.add(tf.keras.layers.Dense(units=64, activation=tf.nn.relu))

# add dropout layer that randomly takes out a % of the layers so training does not become too specilaised
# aka regularisation
model.add(tf.keras.layers.Dropout(rate=0.2))

# add layer that outputs a max value
model.add(tf.keras.layers.Dense(10, activation=tf.nn.softmax))

# complie the model
# using a given optimiser and loss function
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# output summary of layers
log(model.summary())

# -- Train the model
model.fit(train_features, train_labels, epochs=3, verbose=0)

# test the model
# check the accuracy
model.evaluate(test_features, test_labels)

# If you want to save the model in its current state in HDF5 format, you would use the following code syntax:
log("saving...")
with tempfile.NamedTemporaryFile(suffix='.h5', mode='w', delete=False) as f:
    path_object = pathlib.Path(f.name)
    model.save(f.name, overwrite=True, include_optimizer=True)
    save_file(path_object, h5_file_name)

# cleanup
log("cleaning up...")
test_df = None
train_df = None
os.remove(test_file)
os.remove(train_file)

log("complete")