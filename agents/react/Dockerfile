FROM runtime-env

RUN apt-get update && apt-get install -y curl ca-certificates \
    && rm -rf /var/lib/apt/lists/*

RUN curl -LsSf https://astral.sh/uv/install.sh | sh \
    && install -m 755 /root/.local/bin/uv /usr/local/bin/uv

RUN mkdir -p /home/nonroot/.cache/uv && chown -R nonroot:nonroot /home/nonroot/.cache && chmod -R 755 /home/nonroot/.cache

ENV PATH="/root/.local/bin:$PATH"

COPY requirements.txt ${AGENT_DIR}/requirements.txt

RUN mamba run -n ${CONDA_ENV_NAME} pip install -r ${AGENT_DIR}/requirements.txt && \
    conda clean -afy

COPY react-codeagent ${AGENT_DIR}

COPY start.sh ${AGENT_DIR}

COPY config.yaml ${AGENT_DIR}

RUN chown -R nonroot:nonroot /home/agent/src \
    && chmod -R u+rwX /home/agent/src
