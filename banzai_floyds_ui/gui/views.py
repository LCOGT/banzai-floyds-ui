from django.shortcuts import render
from django.views.decorators.http import require_http_methods
from banzai_floyds_ui.gui.forms import LoginForm
from django.views.decorators.csrf import csrf_protect
from django.conf import settings
from rest_framework.response import Response
from rest_framework.decorators import api_view
import requests


def banzai_floyds_view(request, template_name="floyds.html", **kwargs):
    context = {}

    # create some context to send over to Dash:
    dash_context = request.session.get("django_plotly_dash", dict())
    dash_context['auth_token'] = request.session.get('auth_token')
    request.session['django_plotly_dash'] = dash_context

    return render(request, template_name=template_name, context=context)


@csrf_protect
@require_http_methods(["POST"])
def login_view(request):
    form = LoginForm(request.POST)
    if form.is_valid():
        username = form.cleaned_data.get('username')
        password = form.cleaned_data.get('password')
        response = requests.post(settings.OBSPORTAL_AUTH_URL,
                                 data={'username': username,
                                       'password': password})
        if not response.ok:
            form.add_error(None, 'Unable to autheticate. Check your username and password.')
            return render(request, 'floyds.html', {'form': form})
        else:
            request.session['auth_token'] = response.json()['token']
            request.session['django_plotly_dash'] = {'auth_token': request.session['auth_token']}
            request.session['username'] = username
            return render(request, 'floyds.html')


@csrf_protect
@require_http_methods(["POST"])
def logout_view(request):
    if 'auth_token' in request.session:
        del request.session['auth_token']
    if 'auth_token' in request.session['django_plotly_dash']:
        del request.session['django_plotly_dash']['auth_token']

    if 'username' in request.session:
        del request.session['username']
    return render(request, 'floyds.html')


@api_view(['GET'])
def status_view(request):
    return Response({'message': 'BANZAI-FLOYDS service is healthy.'})
