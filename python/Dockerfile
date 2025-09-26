FROM python:3.12-slim

WORKDIR /home/bench
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libsm6 \
    libxext6 \
    build-essential \
    git && \
    rm -rf /var/lib/apt/lists/*

RUN useradd -m -U -s /bin/bash bench

COPY requirements.txt ./

RUN umask 012 # files must not be writeable to the submission script

RUN pip install --no-cache-dir -r requirements.txt

COPY *.py python
COPY ../loaders loaders

ENV PYTHONPATH="/home/bench"

# Avoid potential data leak
# RUN usermod -a -G bench root

# Don't create these folders for now:

#RUN mkdir submission
#RUN mkdir data # it should be readonly
#RUN chown bench:bench submission # submission code can read/write to this dir
#RUN touch submission/__init__.py # should be readonly

CMD [ "python", "python/bench.py" ]
