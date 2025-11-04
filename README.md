# BANZAI-FLOYDS-UI
User interface for re-extraction using BANZAI-FLOYDS

Installation
------------
```
pip install .
```

Deployment
----------
You can run a local development version either by running

```
python banzai_floyds_ui/manage.py migrate
python banzai_floyds_ui/manage.py runserver 8080
```
or via the docker file using
```
docker build -t banzai-floyds-ui
docker run --rm -p 8080:8080 banzai-floyds-ui
```

And then point your browser to [http://127.0.0.1:8080/](http://127.0.0.1:8080/) for the API root of the project.
The main dashboard can be found at [http://127.0.0.1:8080/banzai-floyds](http://127.0.0.1:8080/banzai-floyds).

K8s Deployment
--------------
To run this in a more production-like deployment, you can use skaffold.

```
nix develop --impure
ctlptl apply -f local-cluster.yaml
ctlptl apply -f local-registry.yaml
kubectl create secret generic lco-local-secrets --from-literal=auth-token=${AUTH_TOKEN} --from-literal=aws-access-key-id=${AWS_ACCESS_KEY_ID} --from-literal=aws-secret-access-key=${AWS_SECRET_ACCESS_KEY} --from-literal=banzai-db-address=${BANZAI_DB_ADDRESS}
skaffold -m app dev --port-forward
```
Development
-----------
This package is broken into two subpackages, one for the API, and one for the GUI (web frontend). The API is a standard Django
Rest Framework. The GUI is comprised of a Django Plotly Dash app to enable user interaction.
