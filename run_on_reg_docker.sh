#!/bin/sh

# Run bash in on_reg docker image as a current user with current directory mounted as /data

DOCKER_IMAGE=on_reg

docker run -ti --rm \
    -v `pwd`:/data \
    --user $(id -u):$(id -g) \
    $DOCKER_IMAGE bash
