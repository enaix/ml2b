FROM runtime-env

COPY requirements.txt ${AGENT_DIR}/requirements.txt

RUN pip install -r ${AGENT_DIR}/requirements.txt

COPY . ${AGENT_DIR}

# ENV VERTEX=true
# ENV VERTEX_SERVICE_ACCOUNT=${AGENT_DIR}/service_account.json
