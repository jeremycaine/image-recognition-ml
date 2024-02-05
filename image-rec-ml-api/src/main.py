 #!/usr/bin/env python3

import flask
import os
import io
import pathlib 

import PIL
import sys
import PIL.ImageOps
import numpy as np

from tensorflow import keras
import h5py
import tempfile

import ibm_boto3
from ibm_botocore.client import Config
from ibm_botocore.exceptions import ClientError

app = flask.Flask(__name__)

# credentials
COS_API_KEY_ID = os.getenv('COS_API_KEY_ID')

COS_ENDPOINT = os.getenv('COS_ENDPOINT', 'https://s3.eu-gb.cloud-object-storage.appdomain.cloud')
COS_AUTH_ENDPOINT = os.getenv('COS_AUTH_ENDPOINT', 'https://iam.cloud.ibm.com/identity/token')
COS_SERVICE_CRN = os.getenv('COS_SERVICE_CRN', 'crn:v1:bluemix:public:cloud-object-storage:global:a/b71ac2564ef0b98f1032d189795994dc:875e3790-53c1-40b0-9943-33b010521174::')
COS_STORAGE_CLASS = os.getenv('COS_STORAGE_CLASS','eu-gb-smart')

# file vars
bucket_name = os.getenv('BUCKET_NAME', 'mnist-model')
h5_file_name = os.getenv('H5_FILE_NAME', 'mnist-model.h5')

def log(e):
    print("{0}\n".format(e))

# Retrieve the list of available buckets
def get_buckets():
    print("Retrieving list of buckets")
    try:
        bucket_list = cos_cli.list_buckets()
        for bucket in bucket_list["Buckets"]:
            print("Bucket Name: {0}".format(bucket["Name"]))
        
        log("done")
    except ClientError as be:
        log(be)
        sys.exit(1)
    except Exception as e:
        log("Unable to retrieve list buckets: {0}".format(e))
        sys.exit(1)

def init(): 
    try:
        # Create a temporary file
        with tempfile.NamedTemporaryFile() as f:
            # Get the pathlib.Path object
            path = pathlib.Path(f.name)

            # and file name
            fn = path.name

            cos_cli.download_file(bucket_name, h5_file_name, fn)    
            with h5py.File(fn, 'r') as hdf_file:
                print(hdf_file)
                model = keras.models.load_model(hdf_file, compile=False)
            
            os.remove(fn)
        
        log("Loaded model successfully from COS")
        return model
    except ClientError as be:
        log(be)
        sys.exit(1)
    except Exception as e:
        log("Unable to get file: {0}".format(e))
        sys.exit(1)

log("getting cos connection ...")
# create cloud objec storage connection
cos_cli = ibm_boto3.client("s3",
    ibm_api_key_id=COS_API_KEY_ID,
    ibm_service_instance_id=COS_SERVICE_CRN,
    ibm_auth_endpoint=COS_AUTH_ENDPOINT,
    config=Config(signature_version="oauth"),
    endpoint_url=COS_ENDPOINT
)

global model
model = init()

@app.route('/', methods=['GET'])
def index():
    return "ready"

@app.route('/image', methods=['POST'])
def image():
    # Note, even though we do a little prep, we don't clean the image nearly
    # as well as the MNIST dataset expects, so there will be some issues with
    # generalizing it to handwritten digits

    # Start by taking the image into pillow so we can modify it to fit the
    # right size
    pilimage: PIL.Image.Image = PIL.Image.frombytes(
        mode="RGBA", size=(200, 200), data=flask.request.data)

    # Resize it to the right internal size
    pilimage = pilimage.resize((20, 20))

    # Need to replace the Alpha channel, since 0=black in PIL
    newimg = PIL.Image.new(mode="RGBA", size=(20, 20), color="WHITE")
    newimg.paste(im=pilimage, box=(0, 0), mask=pilimage)

    # Turn it from RGB down to grayscale
    grayscaled_image = PIL.ImageOps.grayscale(newimg)

    # Add the padding so we have it at 28x28, with the 4px padding on all sides
    padded_image = PIL.ImageOps.expand(
        image=grayscaled_image, border=4, fill=255)

    # Call Invert here, since Pillow assumes 0=black 255=white, and our neural
    # net assumes the opposite
    inverted_image = PIL.ImageOps.invert(padded_image)

    # Finally, convert our image to the (28, 28, 1) format expected by the
    # model. Tensorflow expects an array of inputs, so we end up with
    reshaped_image = np.array(
        list(inverted_image.tobytes())).reshape((1, 28, 28, 1))

    scaled_image_array = reshaped_image / 255.0
    log(type(scaled_image_array))

    # now call the model to predict what the digit the image is
    out = model.predict(scaled_image_array)
    log("Predicted Image is : " + str(np.argmax(out,axis=1)))
    
    response = np.array_str(np.argmax(out,axis=1))
    print(type(response))
    return response	
 
# Get the PORT from environment
port = os.getenv('PORT', '8081')
debug = os.getenv('DEBUG', 'false')
if __name__ == "__main__":
    log("application ready - Debug is " + str(debug))
    app.run(host='0.0.0.0', port=int(port), debug=debug)
