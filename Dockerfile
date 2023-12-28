FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update \
    && apt-get install -y \
        nano python3-pip python3-mock libpython3-dev \
        libpython3-all-dev python-is-python3 wget curl cmake \
        software-properties-common sudo git libgl1-mesa-dev \
    && sed -i 's/# set linenumbers/set linenumbers/g' /etc/nanorc \
    && apt clean \
    && rm -rf /var/lib/apt/lists/*

RUN pip install pip -U \
    && pip install \
        --index-url https://download.pytorch.org/whl/cpu \
        torch \
        torchvision \
        torchaudio \
    && pip install \
        onnx==1.15.0 \
        onnxsim==0.4.33 \
        onnxruntime==1.16.1 \
        simple-onnx-processing-tools==1.1.30 \
        Cython==3.0.7 \
        h5py==3.10.0 \
        Pillow==9.3.0 \
        six==1.16.0 \
        scipy==1.11.4 \
        opencv-python==4.8.1.78 \
        matplotlib==3.8.2 \
        tb-nightly==2.16.0a20231227 \
        future==0.18.3 \
        yacs==0.1.8 \
        gdown==4.7.1 \
        flake8==6.1.0 \
        yapf==0.40.2 \
        isort==4.3.21 \
        imageio==2.33.1 \
        chardet==5.2.0 \
    && python -m pip install -U onnx_graphsurgeon \
        --index-url https://pypi.ngc.nvidia.com

RUN git clone https://github.com/KaiyangZhou/deep-person-reid.git \
    && cd deep-person-reid \
    && python setup.py develop

ENV USERNAME=user
RUN echo "root:root" | chpasswd \
    && adduser --disabled-password --gecos "" "${USERNAME}" \
    && echo "${USERNAME}:${USERNAME}" | chpasswd \
    && echo "%${USERNAME}    ALL=(ALL)   NOPASSWD:    ALL" >> /etc/sudoers.d/${USERNAME} \
    && chmod 0440 /etc/sudoers.d/${USERNAME}
USER ${USERNAME}
ARG WKDIR=/workdir
WORKDIR ${WKDIR}
RUN sudo chown ${USERNAME}:${USERNAME} ${WKDIR}