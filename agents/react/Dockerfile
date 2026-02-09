FROM runtime-env

RUN mkdir -p /home/nonroot/.cache/uv \
 && chown -R nonroot:nonroot /home/nonroot/.cache \
 && chmod -R 755 /home/nonroot/.cache

ENV UV_HTTP_TIMEOUT=600

COPY react-codeagent/pyproject.toml react-codeagent/uv.lock ${AGENT_DIR}/

RUN pip compile ${AGENT_DIR}/pyproject.toml -o ${AGENT_DIR}/requirements.txt \
 && pip install --python /opt/venv -r ${AGENT_DIR}/requirements.txt \
 && rm ${AGENT_DIR}/requirements.txt

COPY start.sh config.yaml additional_notes.txt ${AGENT_DIR}/
COPY react-codeagent ${AGENT_DIR}/

RUN chmod +x ${AGENT_DIR}/start.sh