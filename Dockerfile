# syntax=docker/dockerfile:1

FROM ghcr.io/lcogt/banzai-floyds:0.17.1

USER root

RUN apt-get update && apt-get install -y git

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /banzai-floyds-ui

ENV UV_PYTHON_DOWNLOADS=never

RUN --mount=type=cache,target=/root/.cache \
    --mount=type=bind,source=uv.lock,target=uv.lock \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    uv venv --system-site-packages && uv sync --locked --no-dev --no-install-project

COPY . .

RUN --mount=type=cache,target=/root/.cache uv sync --locked --no-dev

ENV PATH="/banzai-floyds-ui/.venv/bin:$PATH"

WORKDIR /banzai-floyds-ui/banzai_floyds_ui/

RUN python manage.py collectstatic

RUN python manage.py createcachetable

RUN python manage.py makemigrations

RUN python manage.py migrate

CMD [ \
    "gunicorn", \
    "--bind=0.0.0.0:8080", \
    "--worker-class=gevent", \
    "--workers=4", \
    "--timeout=300", \
    "--capture-output", \
    "--enable-stdio-inheritance", \
    "--access-logfile=-", \
    "--error-logfile=-", \
    "banzai_floyds_ui.gui.wsgi:application" \
    ]
