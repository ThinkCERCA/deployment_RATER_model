#docker run -it --gpus all --net host --shm-size=256g --name tlab -v /home/crodoc/Documents/tlab:/workspace tlab bash

FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-devel

RUN apt-get update && apt-get upgrade -y

ENV NB_USER jovyan
ENV NB_UID 1000
ENV HOME /home/$NB_USER

RUN useradd -M -s /bin/bash -N -u ${NB_UID} ${NB_USER} \
 && mkdir -p ${HOME} \
 && chown -R ${NB_USER}:users ${HOME} \
 && chown -R ${NB_USER}:users /usr/local/bin

ENV PATH "$PATH:/home/jovyan/.local/bin"

USER ${NB_UID}

RUN pip install pandas
RUN pip install notebook
RUN pip install ipykernel
RUN pip install scikit-learn
RUN pip install pandas
RUN pip install tqdm
RUN pip install transformers
RUN pip install bitsandbytes
RUN pip install pytorch-lightning
RUN pip install sentencepiece
RUN pip install ipywidgets

RUN pip install mmcv==1.4.6
RUN pip uninstall -y opencv-python && pip install opencv-python-headless

RUN pip install text_unidecode
RUN pip install -U protobuf==3.20.1
RUN pip install seaborn
RUN pip install wandb
