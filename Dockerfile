FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

COPY pyproject.toml README.md LICENSE ./
COPY src ./src
RUN pip install .

# run as a normal user, not root
RUN useradd --create-home agent
USER agent
WORKDIR /home/agent

EXPOSE 8765

ENTRYPOINT ["research-agent"]
CMD ["--help"]
