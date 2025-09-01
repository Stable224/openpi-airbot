ARG BASE_IMAGE=registry.qiuzhi.tech/devops/openpi:base

FROM ${BASE_IMAGE}

ENV http_proxy="http://192.168.211.162:10100"

ENV https_proxy="http://192.168.211.162:10100"

ENV no_proxy="localhost,127.0.0.1,::1,172.25.12.10,172.25.0.0/16,192.168.0.0/16"

WORKDIR /openpi

COPY  .  .

RUN GIT_LFS_SKIP_SMUDGE=1 uv sync

RUN pip install packages/dataloop-0.0.18.dev0-py3-none-any.whl -i https://mirrors.aliyun.com/pypi/simple  --break-system-package

RUN apt update && apt install libgl1 -y


RUN GIT_LFS_SKIP_SMUDGE=1 uv pip install -e .
