# BANZAI-FLOYDS-UI
User interface for re-extraction using [BANZAI-FLOYDS](https://github.com/LCOGT/banzai-floyds) 
[[docs](https://github.com/LCOGT/banzai-floyds-ui.git)]

Installation
------------
```
poetry install
```

Deployment
----------
You can run a local development version.

You will want to set a `DB_ADDRESS` environment variable that points to the location of a DB with Banzai Data.
This can be either a local DB with some test data stored in it, or the production DB.

Next,run the following commands to prepare the project and run a local server.

```
python banzai_floyds_ui/manage.py migrate
python banzai_floyds_ui/manage.py createcachetable
python banzai_floyds_ui/manage.py collectstatic
python banzai_floyds_ui/manage.py runserver 8080
```

[//]: # (or via the docker file using)

[//]: # (```)

[//]: # (docker build -t banzai-floyds-ui)

[//]: # (docker run --rm -p 8080:8080 banzai-floyds-ui)

[//]: # (```)

And then point your browser to [http://127.0.0.1:8080/](http://127.0.0.1:8080/) to visit the project dashboard.

K8s Deployment
--------------
To run this in a more production-like deployment, you can use skaffold.

```
nix develop --impure
ctlptl apply -f local-cluster.yaml
ctlptl apply -f local-registry.yaml
kubectl create secret generic lco-local-secrets --from-literal=auth-token=${AUTH_TOKEN} --from-literal=aws-access-key-id=${AWS_ACCESS_KEY_ID} --from-literal=aws-secret-access-key=${AWS_SECRET_ACCESS_KEY} --from-literal=banzai-db-address=${DB_ADDRESS}
skaffold -m app dev --port-forward
```
Development
-----------
This package is broken into two subpackages, one for the API, and one for the GUI (web frontend). The API is a standard Django
Rest Framework. The GUI is comprised of a Django Plotly Dash app to enable user interaction.
