FROM tiangolo/uvicorn-gunicorn:python3.7

RUN pip install starlette
RUN pip install aiofiles
RUN pip install pandas sklearn
RUN pip install gdown
RUN pip install jinja2

COPY ./book2vec /book2vec

WORKDIR /book2vec/models
RUN /book2vec/models/download.sh
WORKDIR /