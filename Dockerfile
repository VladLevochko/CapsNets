FROM nvcr.io/nvidia/pytorch:20.08-py3
COPY data /caps-nets/data
COPY src /caps-nets/src
WORKDIR /caps-nets/src
RUN ls -l