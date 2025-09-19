FROM runtime-env

RUN git clone https://github.com/azrails/ML-Master.git ${AGENT_DIR}

RUN mamba run -n ${CONDA_ENV_NAME} pip install -r ${AGENT_DIR}/requirements.txt && \
    conda clean -afy

COPY . ${AGENT_DIR}
