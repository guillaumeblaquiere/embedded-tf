FROM golang:1.14 as builder

# Copy local code to the container image.
WORKDIR /go/src
COPY . .
RUN GO111MODULE=on CGO_ENABLED=0 GOOS=linux go build -v -o server

FROM ubuntu:xenial

# Install manually TensorflowServing
RUN apt-get update && \
  apt-get install -y curl && \
  echo "deb http://storage.googleapis.com/tensorflow-serving-apt stable tensorflow-model-server tensorflow-model-server-universal" | tee /etc/apt/sources.list.d/tensorflow-serving.list && \
  curl https://storage.googleapis.com/tensorflow-serving-apt/tensorflow-serving.release.pub.gpg | apt-key add - && \
  apt-get update && \
  apt-get install tensorflow-model-server-universal

COPY --from=builder /go/src/server /server
CMD ["/server"]