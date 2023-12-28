FROM nvidia/cuda:11.1.1-cudnn8-devel-ubuntu18.04

ENV DEBIAN_FRONTEND=noninteractive
ENV USERNAME=user

RUN apt-get update \
    && apt-get install -y \
        python3-opencv ca-certificates python3-dev git wget sudo ninja-build cmake nano \
    && sed -i 's/# set linenumbers/set linenumbers/g' /etc/nanorc

RUN ln -sv /usr/bin/python3 /usr/bin/python

# https://github.com/facebookresearch/detectron2/issues/3933
ENV PATH="/home/${USERNAME}/.local/bin:${PATH}"
RUN wget https://bootstrap.pypa.io/pip/3.6/get-pip.py \
    && python3 get-pip.py \
    && rm get-pip.py

# install dependencies
# See https://pytorch.org/ for other options if you use a different version of CUDA
RUN pip install pip -U \
    && pip install \
        setuptools==59.5.0 \
        numpy \
        Cython \
        h5py \
        Pillow \
        six \
        scipy \
        matplotlib \
        tb-nightly \
        future \
        yacs \
        gdown \
        flake8 \
        yapf \
        isort==4.3.21 \
        imageio \
        chardet

RUN pip install \
    torch==1.10 \
    torchvision==0.11.1 \
    -f https://download.pytorch.org/whl/cu111/torch_stable.html

RUN git clone https://github.com/KaiyangZhou/deep-person-reid.git \
    && cd deep-person-reid \
    && apt-get install -y protobuf-compiler libprotobuf-dev \
    && cp /dev/null requirements.txt \
    && python setup.py develop

RUN git clone https://github.com/protocolbuffers/protobuf.git \
    && cd protobuf \
    && git checkout v3.20.2 \
    && git submodule update --init --recursive \
    && mkdir build_source && cd build_source \
    && cmake ../cmake -Dprotobuf_BUILD_SHARED_LIBS=OFF -DCMAKE_INSTALL_PREFIX=/usr -DCMAKE_INSTALL_SYSCONFDIR=/etc -DCMAKE_POSITION_INDEPENDENT_CODE=ON -Dprotobuf_BUILD_TESTS=OFF -DCMAKE_BUILD_TYPE=Release \
    && make -j$(nproc) \
    && make install

RUN git clone -b v1.14.1 https://github.com/onnx/onnx.git \
    && cd onnx \
    && git submodule update --init --recursive \
    && wget https://patch-diff.githubusercontent.com/raw/onnx/onnx/pull/5363.patch \
    && git apply 5363.patch \
    && export CMAKE_ARGS=-DONNX_USE_LITE_PROTO=ON \
    && pip install -e .

RUN pip install onnxruntime

RUN echo "root:root" | chpasswd \
    && adduser --disabled-password --gecos "" "${USERNAME}" \
    && echo "${USERNAME}:${USERNAME}" | chpasswd \
    && echo "%${USERNAME}    ALL=(ALL)   NOPASSWD:    ALL" >> /etc/sudoers.d/${USERNAME} \
    && chmod 0440 /etc/sudoers.d/${USERNAME}
USER ${USERNAME}
ARG WKDIR=/workdir
WORKDIR ${WKDIR}
RUN sudo chown ${USERNAME}:${USERNAME} ${WKDIR}
