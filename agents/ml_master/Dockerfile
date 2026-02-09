FROM runtime-env

RUN git clone https://github.com/azrails/ML-Master.git ${AGENT_DIR}

RUN pip install -r ${AGENT_DIR}/requirements.txt

COPY . ${AGENT_DIR}
