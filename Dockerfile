FROM ghcr.io/lcogt/banzai-floyds:0.11.1

WORKDIR /banzai-floyds-ui

COPY ./pyproject.toml ./dependencies.lock ./

RUN pip install --no-cache -r dependencies.lock

COPY . .

RUN pip install --no-cache-dir .

WORKDIR /banzai-floyds-ui/banzai_floyds_ui

RUN python manage.py collectstatic

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
