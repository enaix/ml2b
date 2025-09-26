FROM runtime-env

COPY requirements.txt ${AGENT_DIR}/requirements.txt

RUN conda run -n ${CONDA_ENV_NAME} pip install -r ${AGENT_DIR}/requirements.txt && \
    conda clean -afy

COPY . ${AGENT_DIR}