FROM python:3.12-slim

# -------------------------------------------------------------------
# WARNING: This dockerfile is used only for manual grading.
#          See environments/runtime/Dockerfile for a proper runtime.
# -------------------------------------------------------------------

WORKDIR /home/bench
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libsm6 \
    libxext6 \
    build-essential \
    git && \
    rm -rf /var/lib/apt/lists/*

# -------------------------------------------------------------------
# Install R (same as in full runtime, except for postintsall)
# WARNING: packages installed by the agent are not handled!
# -------------------------------------------------------------------

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
# Configure python
# -------------------------------------------------------------------

RUN useradd -m -U -s /bin/bash bench

COPY python/requirements.txt ./

RUN umask 012 # files must not be writeable to the submission script

RUN pip install --no-cache-dir -r requirements.txt

COPY python python
COPY loaders loaders

ENV PYTHONPATH="/home/bench"

# Avoid potential data leak
# RUN usermod -a -G bench root

# Don't create these folders for now:

#RUN mkdir submission
#RUN mkdir data # it should be readonly
#RUN chown bench:bench submission # submission code can read/write to this dir
#RUN touch submission/__init__.py # should be readonly
RUN mkdir -p submission && chown -R bench:bench submission
USER bench

# -------------------------------------------------------------------
# R postinstall
# -------------------------------------------------------------------

# Since we're executing the grader from the unpriviledged user, we have to make the R library folder
RUN R -e 'dir.create(Sys.getenv("R_LIBS_USER"), recursive = TRUE)'

CMD [ "python", "python/bench.py" ]
