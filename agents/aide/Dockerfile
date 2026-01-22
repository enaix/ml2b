FROM runtime-env

COPY requirements.txt ${AGENT_DIR}/requirements.txt

RUN mamba run -n ${CONDA_ENV_NAME} pip install -r ${AGENT_DIR}/requirements.txt && \
    conda clean -afy

COPY . ${AGENT_DIR}

# ENV VERTEX=true
# ENV VERTEX_SERVICE_ACCOUNT=${AGENT_DIR}/service_account.json
