"""banzai_floyds_ui URL Configuration

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/4.1/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.contrib import admin
from django.urls import path, include
from banzai_floyds_ui.gui.views import banzai_floyds_view, login_view, logout_view


urlpatterns = [
    path('admin/', admin.site.urls),
    path('', include('banzai_floyds_ui.api.urls')),
    path('banzai-floyds', banzai_floyds_view, name="banzai-floyds"),
    path('login', login_view, name="login"),
    path('logout', logout_view, name="logout"),
    path('django_plotly_dash', include('django_plotly_dash.urls')),
]
