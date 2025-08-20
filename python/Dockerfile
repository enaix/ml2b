FROM python:3.12-slim

WORKDIR /bench

COPY requirements.txt ./

RUN umask 012 # files must not be writeable to the submission script

RUN pip install --no-cache-dir -r requirements.txt

COPY *.py .

RUN useradd -M -U -s /bin/bash bench

# Avoid potential data leak
# RUN usermod -a -G bench root

# Don't create these folders for now:

#RUN mkdir submission
#RUN mkdir data # it should be readonly
#RUN chown bench:bench submission # submission code can read/write to this dir
#RUN touch submission/__init__.py # should be readonly

CMD [ "python", "./bench.py" ]
