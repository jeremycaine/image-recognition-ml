# Image recognition machine learning prediction app
Train and use the MNIST image prediction model.

This project creates and trains a machine learning model that predicts what number a handrawn digit (0-9) is. It is built using the code and data sets of the "hello world" of ML - the MNIST image dataset.

It uses the IBM Cloud environment for object storage of training and model files. The prediction app can be run locally and serverless in the IBM Code Engine environment

The `train-model` folder contains the code to create and train the model. When using serverless, it runs as a Job in IBM Code Engine and stores the output model in IBM Cloud Object Storage (COS). IBM Code Engine serverless environment is pure x86, not GPUs are used for training the model. For MNIST image, the model and training data is small enough that this simple x86 environment works well.

The `image-rec-app` folder contains the code to run a web app where you can handraw a digit (using mouse or trackpad). When using serverless, the web app is an IBM Code Engine web application. 

The `image-rec-ml-api` folder contains the code to serve an API which takes an image as input. When using serverless, this app is an IBM Code Engine web application. The API app loads with an `init()` that downloads the model file from IBM COS and the loads the model into memory.

A serverless environment is useful for simple build and hosting of application for experimentation when you might not have sufficient resources in your local environment. 

> Note: this is for demonstration purposes only and is not really the type of architecture you use in production. Typically after creating the model is served on some ML infrastructure e.g. Seldon is used in the Open Data Hub open source project (which is the upstream OSS for Red Hat OpenShift Data Science).

## MNIST Image Project
The MNIST Image dataset has been provided in an easy-to-use CSV format. You can get them [here](https://www.kaggle.com/datasets/oddrationale/mnist-in-csv).

The original Yann LeCun MNIST database site is [here](http://yann.lecun.com/exdb/mnist/) but the data formatting is considered not for beginners.

The code to build and train this Deep Learning model - a Convolutional Neural Net (CNN) - is taken from [this Red Hat tutorial](https://developers.redhat.com/learn/openshift-data-science/how-create-tensorflow-model)

## Setup
from `image-recognition-ml` directory go to `train-model`, `image-rec-app`, and `image-rec-ml-api` and the `src` directory in each and run `pip install -r requirements.txt`
```
# e.g. if you are using conda to manage your python environment
conda activate py11
cd ~/train-mode/src
pip install -r requirements.txt
```

### Local Apple Mac on Metal
For those using Apple Macbook and M1 chip you need to setup Tensorflow accordingly.
https://developer.apple.com/metal/tensorflow-plugin/

Tested with
```
python -m pip install tensorflow-macos==2.10.0

# tensorflow-macos        2.10.0
# tensorflow-metal        0.6.0
```

### IBM Cloud Object Storage
From your IBM Cloud account you need to create a COS bucket in your COS service and then get the credentials to the bucket. Then ensure you local environment has that key.
```
export COS_API_KEY_ID=xxx
```
This bucket contains the data files to train and test the model.
```
mnist_train.csv
mnist_test.csv
```

### Production server
Production Python apps should use a WSGI HTTP server. The `image-rec-app` uses  [Gunicorn](https://gunicorn.org).

Run and test app with Gunicorn locally.
```
cd image-rec-app
pip install gunicorn

# -w for number of worker process numbers; 1 is fine for local dev
# -b for address:port to listen on i.e. browser connects to this, and gunicorn routes to the port the app listens on
# main:app where main is the main.py program and app is the entrypoint name of the application
gunicorn -w 1 -b :3000 main:app
```
In production we can scale the number of containers. In each container there will be a number of Gunicorn worker threads running. This could there be an environment variable injected at deployment time.

## Run as Containers on Local
When you run the containers you need to inject the Cloud Object storage API key to the app running in the container can access the bucket.

Assuming you have full control over your development then a simple hack is to run as root so the container process can write to the file system when it is getting the model file.

### Build the containers
```
cd image-rec-ml-api
podman build -t image-rec-ml-api:latest .
cd image-rec-app
podman build -t image-rec-app:latest .
```

### Run the system
Create a pod for the web app and API to run in. 

We expose the port that the app will listen on for browser requests. 

The API container needs the COS key and to run as root since it is using the local filesystem for temporary files used to get the model from COS and load into memory.
```
podman pod create --name image-rec-pod -p 3000:8080
podman run --pod image-rec-pod -d -u root --name ml-api -e COS_API_KEY_ID=xxxx -t image-rec-ml-api:latest
podman run --pod image-rec-pod -d --name rec-app -e PREDICT_API_URL=http://localhost:8081/image -t image-rec-app:latest
```
Now in a browser go to the app at `http://localhost:3000`. Draw an image in the canvas with your mousepad and Send.


## Run on IBM Cloud Code Engine
Login and set right targets for IBM Code Engine work
```
ibmcloud login  ...

ibmcloud target -r us-south
ibmcloud target -g <group name>
```

### Setup IBM Cloud Engine project
One-time to create an IBM Code Engine project
```
ibmcloud ce project create --name image-recognition-ml
```

#### Configure Project
You will create a secret key for your COS bucket credentials. Then you setup various environment variables in a ConfigMap for you Code Engine project environment. This Secret and the ConfigMap is used in the build of your applications and jobs in Code Engine. If you look in the source code you will see these same parameters defaulted so strictly speaking you don't need them in the ConfigMap. 

```
# check project is there and select it before using it
ibmcloud ce project list
ibmcloud ce project select --name image-recognition-ml

# need secret
ibmcloud ce secret create --name caine-cos-api-key --from-literal COS_API_KEY_ID=xxxx

# config map for variables
ibmcloud ce configmap create --name image-recognition-ml \
    --from-literal COS_ENDPOINT=https://s3.eu-gb.cloud-object-storage.appdomain.cloud  \
    --from-literal COS_AUTH_ENDPOINT=https://iam.cloud.ibm.com/identity/token  \
    --from-literal COS_SERVICE_CRN=crn:v1:bluemix:public:cloud-object-storage:global:a/b71ac2564ef0b98f1032d189795994dc:875e3790-53c1-40b0-9943-33b010521174::  \
    --from-literal COS_STORAGE_CLASS=eu-gb-smart  \
    --from-literal H5_FILE_NAME=mnist-model.h5  \
    --from-literal TRAIN_CSV=mnist_train.csv  \
    --from-literal TEST_CSV=mnist_test.csv
```

#### Train Image Prediction Model
Job to train image prediction model
```
# create app first time
ibmcloud ce job create --name train-model --src https://github.com/jeremycaine/image-recognition-ml --bcdr train-model --str dockerfile --env-from-secret caine-cos-api-key --env-from-configmap image-recognition-ml --size large

# or, rebuild after git commit
ibmcloud ce job update --name train-model --rebuild

# when you want to delete it
ibmcloud ce job delete --name train-model
```

#### Predict Image Model API 
API receives image and uses model to predict what number (0-9) the handdrawn digit is in the image
```
# create API first time
ibmcloud ce app create --name image-rec-ml-api --src https://github.com/jeremycaine/image-recognition-ml --bcdr image-rec-ml-api --str dockerfile --env-from-secret caine-cos-api-key --env-from-configmap image-recognition-ml

# or, rebuild after git commit
ibmcloud ce app update --name image-rec-ml-api --rebuild
```

##### Get the API app URL
The create command call completes returning the URL of the API app
e.g. `https://image-rec-ml-api.1d0xljwi5e8d.us-south.codeengine.appdomain.cloud`

The API now has a URL in the serverless environment. This needs to be in an environment variable so the web app can use it to call the API.

The PREDICT_API_URL is the URL location of the API app plus `/image`.

```
# check the "URL:" line in the details of the app
ibmcloud ce application get -n image-rec-ml-api

# set the env var the image recognition app
ibmcloud ce configmap update --name image-recognition-ml \
    --from-literal PREDICT_API_URL=https://image-rec-ml-api.1d0xljwi5e8d.us-south.codeengine.appdomain.cloud/image 
```

#### Image Prediction App 
App to draw a digit and call the model to predict what the number (0-9) is 
```
# create app first time
ibmcloud ce app create --name image-rec-app --src https://github.com/jeremycaine/image-recognition-ml --bcdr image-rec-app --str dockerfile --env-from-configmap image-recognition-ml

# or, rebuild after git commit
ibmcloud ce app update --name image-rec-app --rebuild
```

### Test the Application in IBM Code Engine
First, create and train the model in IBM Code Engine

From the cloud console and the job `train-model` submit the job. You can watch its progress from Logging in the drop down menu top right.

Then, check that the model file `mnist-model.h5` is in your COS bucket.

Next, you can launch the image recognition web app. In your IBM Code Engine project go to application `image-rec-app`, click on Test Application and you will see link to 'Application URL'. 

You can see it in action [here](https://image-rec-app.1d0xljwi5e8d.us-south.codeengine.appdomain.cloud).

To use the app, use your trackpad or mouse to hand draw a digit between 0 and 9, then click Send. The model is called and a prediction as to what digit you drew comes bacl. Hit Clear to start again.

> Remember this is a serverless environment. If you don't use your app for some time then IBM Code Engine will de-stage it. Then when you go to the URL and demand the app it will get loaded, and in that process it is reloading the model into memory so that the prediction call can be made on it.


