# Two stage build

# Stage 1: builder image
FROM registry.access.redhat.com/ubi8/python-39 

## Add application sources
WORKDIR /app
COPY src/requirements.txt ./

# Install the dependencies
RUN pip install -U "pip>=19.3.1" && \
    pip install tensorflow \
    pip install --no-cache-dir -r requirements.txt

## 2. Non-root, arbitrary user IDs
USER 1001

## 3. Image identification
LABEL name="jeremycaine/mnist-image-ibm-ce/digit-image" \
      vendor="Acme, Inc." \
      version="1.2.3" \
      release="45" \
      summary="Digital image recognition" \
      description="Digit image recognition machine learning app using MNIST image dataset"

USER root

## 4. Image license
## Red Hat requires that the image store the license file(s) in the /licenses directory. 
## Store the license in the image to self-document
COPY ./licenses /licenses

## 5. Latest security updates
RUN dnf clean all

## 6. Group ownership and file permission
RUN chgrp -R 0 $HOME && \
    chmod -R g=u $HOME

USER 1001
RUN chown -R 1001:0 $HOME

## 7. Application source
## Copy the application source and build artifacts from the builder image to this one
COPY src/ ./

## Set environment environment variables and expose port
ENV PORT 8080
EXPOSE 8080

# Gunicorn https://docs.gunicorn.org/en/latest/design.html#how-many-workers
ENV WORKERS 3

## Run script uses standard ways to run the application
CMD gunicorn -w ${WORKERS} -b :${PORT} main:app