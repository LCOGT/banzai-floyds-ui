FROM ghcr.io/lcogt/banzai-floyds:0.17.1

USER root

RUN poetry config virtualenvs.create false

COPY pyproject.toml poetry.lock /banzai-floyds-ui/

RUN poetry install --directory=/banzai-floyds-ui --no-root --no-cache

COPY . /banzai-floyds-ui

RUN poetry install --directory /banzai-floyds-ui --no-cache

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
