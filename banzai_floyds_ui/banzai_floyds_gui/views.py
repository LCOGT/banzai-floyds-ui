from django.shortcuts import render
from django.views.decorators.http import require_http_methods
from django.contrib.auth.forms import AuthenticationForm
from django.conf import settings
import requests


def banzai_floyds_view(request, template_name="floyds.html", **kwargs):
    context = {}

    # create some context to send over to Dash:
    dash_context = request.session.get("django_plotly_dash", dict())
    dash_context['auth_token'] = request.session.get('auth_token')
    request.session['django_plotly_dash'] = dash_context

    return render(request, template_name=template_name, context=context)


@require_http_methods(["POST"])
def login_view(request):
    form = AuthenticationForm(request, data=request.POST)
    if form.is_valid():
        username = form.cleaned_data.get('username')
        password = form.cleaned_data.get('password')
        response = requests.post(settings.OBSPORTAL_AUTH_URL,
                                 data={'username': username,
                                       'password': password})
        if response.status_code != requests.HTTPStatus.OK:
            form.add_error(None, 'Unable to autheticate. Check your username and password.')
            return render(request, 'floyds.html', {'form': form})
        else:
            request.session['auth_token'] = response.json()['token']
            request.session['username'] = username
            return render(request, 'floyds.html')
