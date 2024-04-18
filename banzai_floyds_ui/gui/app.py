import dash
from dash import dcc, html
from django_plotly_dash import DjangoDash
import requests
from django.conf import settings
import logging
import io
from astropy.io import fits
import plotly.graph_objs as go
import datetime
import httpx
import asyncio


logger = logging.getLogger(__name__)

dashboard_name = 'banzai-floyds'
app = DjangoDash(name=dashboard_name)


def layout():
    last_week = datetime.date.today() - datetime.timedelta(days=7)
    return html.Div(
        id='main',
        children=[
            html.Div(id='options-container', children=[
                html.Div(children=[dcc.DatePickerRange(id='date-range-picker',
                                                       start_date=last_week.strftime('%Y-%m-%d'),
                                                       end_date=datetime.date.today().strftime('%Y-%m-%d')),
                                   dcc.Dropdown(id='site-picker', options=[{'label': 'ogg', 'value': 'en06'},
                                                                           {'label': 'coj', 'value': 'en12'}],
                                                value=None, placeholder='Site',
                                                style={'float': 'left', 'font-size': '1.1875rem',
                                                       'min-width': '7rem', 'height': '3rem'})]),
                dcc.Loading(id='loading-dropdown', type='default',
                            children=[dcc.Dropdown(id='file-list-dropdown', options=[], placeholder='Select spectrum',
                                                   value=None, className='col-md-12')])]),
            dcc.Loading(id='loading-image-plot', type='default', children=[html.Div(id='image-plot-output-div')])]
    )


app.layout = layout


async def fetch(url, params, headers):
    async with httpx.AsyncClient() as client:
        response = await client.get(url, params=params, headers=headers)
    return response


async def fetch_all(archive_header, request_params):
    tasks = [fetch(settings.ARCHIVE_URL, params, archive_header) for params in request_params]
    return await asyncio.gather(*tasks)


@app.expanded_callback(dash.dependencies.Output('file-list-dropdown', 'options'),
                       [dash.dependencies.Input('date-range-picker', 'start_date'),
                        dash.dependencies.Input('date-range-picker', 'end_date'),
                        dash.dependencies.Input('site-picker', 'value')])
def callback_dropdown_files(*args, **kwargs):
    start_date, end_date = args[0:2]

    # Loop over sites to unless one is chosen
    instrument_id = args[2]
    if instrument_id is None:
        instrument_ids = ['en06', 'en12']
    else:
        instrument_ids = [instrument_id]

    if kwargs['session_state'].get('auth_token') is not None:
        archive_header = {'Authorization': f'Token {kwargs["session_state"]["auth_token"]}'}
    else:
        archive_header = None

    request_params = [{'start': start_date, 'end': end_date, 'public': True, 'limit': 150,
                       'instrument_id': instrument_id, 'RLEVEL': 91}
                      for instrument_id in instrument_ids]

    responses = asyncio.run(fetch_all(archive_header, request_params))
    results = []
    for response in responses:
        try:
            response.raise_for_status()
        except requests.exceptions.HTTPError as e:
            logger.error(f"Failed to fetch data from archive: {e}. {response.content}")
            return
        data = response.json()['results']
        results += [{'label': row['filename'], 'value': row['id']} for row in data]
    results.sort(key=lambda x: x['label'])
    return results


@app.expanded_callback(
    dash.dependencies.Output('image-plot-output-div', 'children'),
    dash.dependencies.Input('file-list-dropdown', 'value'))
def callback_image_plot(*args, **kwargs):
    file_to_plot = args[0]
    if file_to_plot is None:
        return None
    if kwargs['session_state'].get('auth_token') is not None:
        archive_header = {'Authorization': f'Token {kwargs["session_state"]["auth_token"]}'}
    else:
        archive_header = None
    response = requests.get(f'{settings.ARCHIVE_URL}{args[0]}', headers=archive_header)
    response.raise_for_status()
    buffer = io.BytesIO()
    buffer.write(requests.get(response.json()['url'], stream=True).content)
    buffer.seek(0)

    hdu = fits.open(buffer)
    # Create a trace
    trace = go.Heatmap(z=hdu['SCI'].data)

    layout = dict(title='', margin=dict(t=20, b=50, l=50, r=40), height=350)

    fig = dict(data=[trace], layout=layout)
    image_plot = dcc.Graph(id='image-graph1', figure=fig, style={'display': 'inline-block',
                                                                 'width': '100%', 'height': '100%;'})
    # Return the plot as the child of the output container div
    return [image_plot]
