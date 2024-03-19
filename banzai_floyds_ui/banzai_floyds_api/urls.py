from django.urls import path
from banzai_floyds_ui.banzai_floyds_api import views

urlpatterns = [
    path('', views.printHelp)
]
