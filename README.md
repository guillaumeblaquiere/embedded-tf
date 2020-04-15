# Overview

Embedded-tf performs batch prediction by loading, at runtime, Tensorflow model and starting a Tensorflow server on each request.
The container is designed for running on Cloud Run, but can be deployed anywhere (on GKE, on K8S, on VM).

With only one deployment, you can perform batch prediction on dynamically loaded model. 

*Related [Medium article](https://medium.com/google-cloud/on-demand-small-batch-predictions-with-cloud-run-and-embedded-tf-469242d66c3b
)*

## Internal steps

The container exposes a web server which, on each request:

* Downloads the model to use from GCS bucket
* Starts a Tensorflow server with the loaded model
* For each input file
  * Download the input file in memory
  * Format the input file in the Tensorflow server expected JSON format
  * Perform the prediction and get the body response
  * Format the body response for having a JSON line output
  * Upload the output into the bucket/path output
* Kill Tensorflow server and clean the local data

The output file hierarchy follows the input file hierarchy  

## Caveats

### Memory size
The container stores the files in `/tmp` directory. 

On managed Cloud Run, it's an in-memory file system. Take care of the memory footprint:
* The model files are stored in `/tmp` directory (in-memory file system).
* The input and output file content are kept in a variable in memory.
* The app, including the Go web server, the ephemeral Tensorflow server which its request and its response has impact on memory.
 
It mustn't exceed the total memory allowed on the service (max 2Gb with Cloud Run managed).

### Concurrency
The container is able to handle only one request at the time (because of previous point). 

Set the `concurrency` param to 1 when you deploy and use this container.

# How to install

Here a Cloud Run deployment example
```
gcloud deploy run <SERVICE_NAME> \
--region=<REGION_ID> \
--memory=<MEMORY_SIZE>
--platform=managed \
--image gcr.io/embedded-tf/embedded-tf \
--concurrency=1
```
* **SERVICE_NAME**: set the name on the service you want
* **REGION_ID**: set on of the [Cloud Run available region](https://cloud.google.com/run/docs/locations)
* **MEMORY_SIZE**: set the correct [size of memory](https://cloud.google.com/run/docs/configuring/memory-limits) according with your model, input and output size

# How to request

There is 3 required query parameters when you call your deployment

* **model**: GCS location of your model version. Must start by `gs://`. The root path must contain the `.pb` files and variables. Example `gs://mybucket/mymodel/export/exporter/1546446862/`
* **input**: GCS location of your input file(s). Must start by `gs://`. 
  * If the param end with `/`, all the files and subdirectories are downloaded and used as input. 
  * Else, the unique referenced file is downloaded and used as input.
* **output**: GCS location where the prediction are uploaded. Must start by `gs://`. The path defines a GCS directory.

A typical call is the following
```
curl -H "Authorization: $(gcloud auth print-identity-token)" \
"https://<SERVICE_NAME>-<project hash and region>.run.app?model=<MODEL_PATH>&input=<INPUT_PATH>&output=<OUTPUT_PATH>" 
```

## File format

The data format is the same as [AI Platform batch prediction](https://cloud.google.com/ai-platform/prediction/docs/batch-predict#configuring_a_batch_prediction_job)
**only in JSON format**

In summary, the format is a JSON line, with 1 valid JSON object on each line (no indentation).

* One instance to predict per line in the input files
* One prediction result per line in the output files

# Build the container

If you want to rebuild yourself the container, a [Cloud Build](https://github.com/guillaumeblaquiere/embedded-tf/tree/master/cloudbuild.yaml)
 file is present in addition of a [Dockerfile](https://github.com/guillaumeblaquiere/embedded-tf/tree/master/Dockerfile)

```
# Cloud Build with yaml configuration
gcloud builds submit

# Cloud Build with only the Dockerfile
gcloud builds submit -t gcr.io/<PROJECT_ID>/<container_name>

# With local Docker
docker build .
```

# License

This repository is licensed under Apache 2.0. Full license text is available in
[LICENSE](https://github.com/guillaumeblaquiere/embedded-tf/tree/master/LICENSE).
