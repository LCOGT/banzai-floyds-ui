from django.urls import path
from banzai_floyds_ui.api import views

urlpatterns = [
    path('', views.printHelp)
]
