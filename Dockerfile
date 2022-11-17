FROM nvidia/cuda:11.3.0-devel-ubuntu20.04
LABEL Author="Linghai Wang"

# Install base utilities
# RUN rm /etc/apt/sources.list.d/cuda.list
# RUN rm /etc/apt/sources.list.d/nvidia-ml.list
RUN apt-get update 
RUN apt-get install -y --no-install-recommends apt-utils
RUN apt-get install -y wget && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*


RUN apt-get update && apt-get -y install git

# Install miniconda
ENV CONDA_DIR /opt/conda
RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh && \
     /bin/bash ~/miniconda.sh -b -p /opt/conda

# Put conda in path so we can use conda activate
ENV PATH=$CONDA_DIR/bin:$PATH

#SHELL ["/bin/bash", "--login", "-c"]
COPY environment.yml environment.yml
RUN set -x \
    && conda init bash \
    && . /root/.bashrc \
    && conda activate base
RUN pip3 install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu116
RUN pip3 install pandas \
    numpy jupyterlab matplotlib \ 
    networkx plotly scikit-learn \
    shap imbalanced-learn \
    pip install tensorboard \
    openpyxl xgboost xlrd nibabel \
    nibabel markdown pdoc3 \
    pytorch-lightning "ray[default]" \
    treelib monai torchio "ray[tune]" \
    pip install GPUtil batchgenerators \
    hydra-core einops pyrsistent
RUN pip3 install git+https://github.com/eduardojdiniz/radio
RUN . /root/.bashrc
ENV PATH /opt/conda/bin:$PATH

COPY test.sh test.sh
COPY pytorch_test.py pytorch_test.py 
RUN chmod +x test.sh

#RUN conda env update -qf environment.yml

