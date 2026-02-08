FROM debian:12-slim

ENV DEBIAN_FRONTEND=noninteractive

# -------------------------------------------------------------------
# system deps
# -------------------------------------------------------------------
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    ca-certificates \
    build-essential \
    libgomp1 \
    libstdc++6 \
    zip \
    unzip \
    ffmpeg \
    libsm6 \
    libxext6 \
    && rm -rf /var/lib/apt/lists/*

# -------------------------------------------------------------------
# uv (правильный путь: /root/.local/bin)
# -------------------------------------------------------------------
ARG PYTHON_VERSION=3.12
ARG REQUIREMENTS=/tmp/requirements.txt

COPY /environments/runtime/requirements.txt ${REQUIREMENTS}

RUN curl -LsSf https://astral.sh/uv/install.sh | sh \
 && export PATH="/root/.local/bin:$PATH" \
 && uv python install ${PYTHON_VERSION} \
 && uv venv /opt/venv --python ${PYTHON_VERSION} \
 && uv pip install --python /opt/venv -r ${REQUIREMENTS} \
 && rm ${REQUIREMENTS} \
 && mv /root/.local/bin/uv /usr/local/bin/uv \
 && mv /root/.local/bin/uvx /usr/local/bin/uvx

# -------------------------------------------------------------------
# pip wrapper для uv
# -------------------------------------------------------------------
RUN printf '#!/bin/bash\nexec uv pip "$@"\n' > /usr/local/bin/pip \
 && chmod +x /usr/local/bin/pip \
 && ln -sf /usr/local/bin/pip /usr/local/bin/pip3

# -------------------------------------------------------------------
# environment
# -------------------------------------------------------------------
ENV PATH="/opt/venv/bin:$PATH" \
    VIRTUAL_ENV="/opt/venv" \
    UV_HTTP_TIMEOUT=600

# -------------------------------------------------------------------
# users & directories
# -------------------------------------------------------------------
ARG AGENT_DIR=/home/agent
ENV AGENT_DIR=${AGENT_DIR}

RUN useradd -r -s /bin/false nonroot \
 && useradd -m -U -s /bin/bash bench \
 && mkdir -p \
        /home/logs \
        /home/submission \
        /home/artifacts \
        /home/data \
        ${AGENT_DIR} \
        /root/.cache/uv \
        /home/nonroot/.cache/uv \
 && chown -R nonroot:nonroot \
        /home/logs \
        /home/submission \
        /home/artifacts \
        /home/data \
        ${AGENT_DIR} \
        /home/nonroot/.cache \
 && chown -R bench:bench /opt/venv \
 && chmod -R 755 /root/.cache /home/nonroot/.cache

WORKDIR /home/bench
COPY --chown=bench:bench python ./python
COPY --chown=bench:bench loaders ./loaders

WORKDIR /home

ENTRYPOINT ["tail", "-f", "/dev/null"]