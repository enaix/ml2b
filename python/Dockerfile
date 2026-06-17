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
