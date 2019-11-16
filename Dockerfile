FROM python as build

WORKDIR /build
COPY pypkg/ .
RUN python3 setup.py sdist bdist_wheel


FROM tensorflow/tensorflow:2.0.0-py3-jupyter
WORKDIR /app
ARG APP_VERSION=latest
RUN echo ${APP_VERSION} > /app/version

RUN apt-get update -yqq && \
    apt-get install jq -yq && \
    apt-get clean -y && \
    rm -rf /var/lib/apt/lists/*

ARG K8S_VERSION=1.14.7

ADD https://storage.googleapis.com/kubernetes-release/release/v${K8S_VERSION}/bin/linux/amd64/kubectl /usr/local/bin/kubectl
RUN chmod +x /usr/local/bin/kubectl && \
    export PATH=/usr/local/bin/:$PATH

COPY --from=build /build/dist/pylib-1.0.0-py3-none-any.whl pylib/pylib-1.0.0-py3-none-any.whl
COPY requirements.txt /app/
RUN pip install --no-cache-dir pylib/pylib-1.0.0-py3-none-any.whl && \
    pip install --no-cache-dir -r /app/requirements.txt && \
    rm -rf pylib /app/requirements.txt

COPY app/*.py /app/
COPY app/evaluate.ipynb /app
COPY makefile /app
COPY run_e2e.sh /
RUN chmod +x /run_e2e.sh /app/*.py
ENV PATH /app/:$PATH

EXPOSE 5000
ENV MODEL_NAME PetSetModel
CMD exec seldon-core-microservice $MODEL_NAME REST --service-type MODEL --persistence 0

