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
# install R
# -------------------------------------------------------------------

# Base instructions taken from https://cran.r-project.org/bin/linux/debian/
# Packages taken from https://github.com/rocker-org/rocker/blob/master/r-base/4.6.0/Dockerfile
# This installs a whatever version of R
RUN apt-get install -y --no-install-recommends \
    libopenblas0-pthread \
    littler \
    r-cran-littler \
    r-base \
    r-base-dev \
    r-base-core \
    r-recommended \
    && rm -rf /tmp/downloaded_packages/ /tmp/*.rds \
    && rm -rf /var/lib/apt/lists/*

# We assume that the base library is writable, and the agent is executed from root
# R-devel is not installed here
# No cuda support

# -------------------------------------------------------------------
# install Julia
# -------------------------------------------------------------------

# https://github.com/docker-library/julia/blob/master/stable/bookworm/Dockerfile

ENV JULIA_PATH /usr/local/julia
ENV PATH $JULIA_PATH/bin:$PATH
ENV JULIA_GPG 3673DF529D9049477F76B37566E3C7DC03D6E495
ENV JULIA_VERSION 1.12.6
RUN set -eux; \
	\
	savedAptMark="$(apt-mark showmanual)"; \
	apt-get update; \
	apt-get install -y --no-install-recommends \
		gnupg \
	; \
	rm -rf /var/lib/apt/lists/*; \
	\
# https://julialang.org/downloads/#julia-command-line-version
# https://julialang-s3.julialang.org/bin/checksums/julia-1.12.6.sha256
	arch="$(dpkg --print-architecture)"; \
	case "$arch" in \
		'amd64') \
			url='https://julialang-s3.julialang.org/bin/linux/x64/1.12/julia-1.12.6-linux-x86_64.tar.gz'; \
			sha256='bbabf3bef19421a9dbd24a767d807606ab85e444323b5a1c73ffe293fa3d079a'; \
			;; \
		'i386') \
			url='https://julialang-s3.julialang.org/bin/linux/x86/1.12/julia-1.12.6-linux-i686.tar.gz'; \
			sha256='2ab43d56adfe96c7b0b9ddab0e049f54f49df24049ec8b482c26737c42af28cd'; \
			;; \
		'arm64') \
			url='https://julialang-s3.julialang.org/bin/linux/aarch64/1.12/julia-1.12.6-linux-aarch64.tar.gz'; \
			sha256='029b93b857bd0ffd627f9a8580d3bbaa63daf008d7b7aed02fbceb8fd57c4899'; \
			;; \
		*) \
			echo >&2 "error: current architecture ($arch) does not have a corresponding Julia binary release"; \
			exit 1; \
			;; \
	esac; \
	\
	curl -fL -o julia.tar.gz.asc "$url.asc"; \
	curl -fL -o julia.tar.gz "$url"; \
	\
	echo "$sha256 *julia.tar.gz" | sha256sum --strict --check -; \
	\
	export GNUPGHOME="$(mktemp -d)"; \
	gpg --batch --keyserver keyserver.ubuntu.com --recv-keys "$JULIA_GPG"; \
	gpg --batch --verify julia.tar.gz.asc julia.tar.gz; \
	gpgconf --kill all; \
	rm -rf "$GNUPGHOME" julia.tar.gz.asc; \
	\
	mkdir "$JULIA_PATH"; \
	tar -xzf julia.tar.gz -C "$JULIA_PATH" --strip-components 1; \
	rm julia.tar.gz; \
	\
# Disable autoremove for now
#   apt-mark auto '.*' > /dev/null; \
#   [ -z "$savedAptMark" ] || apt-mark manual $savedAptMark; \
#   apt-get purge -y --auto-remove -o APT::AutoRemove::RecommendsImportant=false; \
	\
# smoke test
	julia --version

# -------------------------------------------------------------------
# uv (correct path: /root/.local/bin)
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
# pip wrapper for uv
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
