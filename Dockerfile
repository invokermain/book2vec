FROM python:3.7

RUN pip install --no-cache-dir uvicorn gunicorn
RUN pip install starlette
RUN pip install pandas sklearn


RUN ./book2vec/models/download.sh
COPY ./book2vec /book2vec

EXPOSE 80


CMD gunicorn -k "uvicorn.workers.UvicornWorker" book2vec:app