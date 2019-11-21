FROM ubuntu:16.04

RUN apt -y update
RUN apt -y upgrade
RUN apt -y update

# install ANTs 2.2.0; not tested with latest ANTs
RUN apt install -y cmake
RUN apt install -y wget g++ git libz-dev

RUN mkdir -p /src
WORKDIR /src
RUN wget -q https://github.com/ANTsX/ANTs/archive/v2.2.0.tar.gz && \
    tar -zxf v2.2.0.tar.gz && \
    rm v2.2.0.tar.gz
RUN cd ANTs-2.2.0 && \
    mkdir -p build && \
    cd build && \
    cmake .. && \
    make

ENV ANTSPATH=/usr/local/ants
RUN mkdir -p $ANTSPATH && \
    mv /src/ANTs-2.2.0/build/bin/* $ANTSPATH && \
    mv /src/ANTs-2.2.0/Scripts/* $ANTSPATH

# install Python 2.7
RUN apt install -y python python-pip
RUN pip install -U pip
RUN pip install numpy scipy nibabel

ENV ONPATH=/usr/local/on_reg
RUN mkdir -p $ONPATH
WORKDIR $ONPATH
RUN wget -q http://code.activestate.com/recipes/117228-priority-dictionary/download/1/ -O priodict.py

# install fsl
RUN wget -q http://fsl.fmrib.ox.ac.uk/fsldownloads/fslinstaller.py
RUN python fslinstaller.py -d /usr/local/fsl -q -V 6.0.1

ENV FSLDIR=/usr/local/fsl
ENV FSLTCLSH=$FSLDIR/bin/fsltclsh
ENV FSLWISH=$FSLDIR/bin/fslwish
ENV FSLOUTPUTTYPE=NIFTI_GZ
ENV PATH=$PATH:$ANTSPATH:$ONPATH:$FSLDIR/bin

RUN rm /src -rf
RUN apt clean
RUN apt autoremove --purge

COPY * ./

RUN chmod ugo+x on_reg.py on_2d.py on_create_center_from_model.py on_rigid_reg.py prepare_dwi_files.py extract_b0_dwi_mean.py on_centerline.py

VOLUME ["/data"]
WORKDIR /data

CMD bash

