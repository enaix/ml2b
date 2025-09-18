FROM ubuntu:22.04

#base system deps
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    zip \
    unzip \
    sudo \
    python3 \
    python3-pip \
    python3-venv \
    python3-dev \
    python-is-python3 \
    ffmpeg \
    libsm6 \
    libxext6 \
    build-essential \
    git \
    curl \
    wget && \
    rm -rf /var/lib/apt/lists/*

#python runtime deps
RUN wget -O /tmp/Miniforge3.sh "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-$(uname)-$(uname -m).sh" \
    && bash /tmp/Miniforge3.sh -b -p "/opt/conda" \
    && rm /tmp/Miniforge3.sh

ENV PATH="/opt/conda/bin:$PATH"

ARG CONDA_ENV_NAME=agent
ARG PYTHON_VERSION=3.12
ARG REQUIREMENTS=/tmp/requirements.txt

COPY /environments/runtime/requirements.txt ${REQUIREMENTS}
RUN conda create -n ${CONDA_ENV_NAME} python=${PYTHON_VERSION} -y

RUN mamba run -n ${CONDA_ENV_NAME} pip install -r /tmp/requirements.txt && \
    mamba clean -afy

#julia runtime
RUN curl -fsSL https://install.julialang.org | sh -s -- -y
ENV PATH="/root/.juliaup/bin:$PATH"
RUN juliaup add 1.6 && juliaup default 1.6
#как-то тоже можно настроить env для джулии, пока хз

#R runtime
RUN apt-get update && apt-get install -y r-base && rm -rf /var/lib/apt/lists/*

#R deps example
RUN R -e "install.packages(c('methods','jsonlite','tseries'), dependencies=TRUE)"

ARG AGENT_DIR=/home/agent
ENV AGENT_DIR=${AGENT_DIR}
ENV CONDA_ENV_NAME=${CONDA_ENV_NAME}

RUN useradd nonroot \
    && mkdir -p /home/logs /home/submission /home/artifacts /home/data ${AGENT_DIR} \
    && chown -R nonroot:nonroot /home/logs /home/submission /home/artifacts /home/data ${AGENT_DIR}

#add bench user
RUN useradd -m -U -s /bin/bash bench
WORKDIR /home/bench
COPY --chown=bench:bench python ./python
COPY --chown=bench:bench loaders ./loaders

WORKDIR /home

ENTRYPOINT ["tail", "-f", "/dev/null"]