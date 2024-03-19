import dash
from dash import dcc, html
from django_plotly_dash import DjangoDash
import requests
from django.conf import settings
import logging
import io
from astropy.io import fits
import plotly.graph_objs as go


logger = logging.getLogger(__name__)

dashboard_name = 'banzai-floyds'
app = DjangoDash(name=dashboard_name)

# We can define the layout as a function to set the start and end dates on page load

app.layout = html.Div(
    id='main',
    children=[
        html.Div(
            [dcc.DatePickerRange(id='date-range-picker', start_date='2024-01-01',
                                 end_date='2024-05-31'),
             dcc.Loading(id='loading-dropdown', type='default',
                         children=[dcc.Dropdown(id='file-list-dropdown',
                                                options=[],
                                                value=None,
                                                className='col-md-12')])]
        ),
        dcc.Loading(id='loading-image-plot', type='default', children=[html.Div(id='image-plot-output-div')])
    ]
)


@app.expanded_callback(dash.dependencies.Output('file-list-dropdown', 'options'),
                       [dash.dependencies.Input('date-range-picker', 'start_date'),
                        dash.dependencies.Input('date-range-picker', 'end_date')])
def callback_dropdown_files(*args, **kwargs):
    # pylint: disable=unused-argument
    'Callback to generate test data on each change of the dropdown'
    # Probably want to use the dash callback loading css here
    # https://dash.plotly.com/loading-states
    # See also: https://codepen.io/chriddyp/pen/brPBPO
    response = requests.get(settings.ARCHIVE_URL,
                            params={'start': args[0],
                                    'end': args[1],
                                    'public': True,
                                    'limit': 150,
                                    'instrument_id': 'en06',
                                    'RLEVEL': 91})
    logger.info(response)
    # TODO: Show a loading spinner while the request is being made
    response.raise_for_status()
    data = response.json()['results']
    return [{'label': row['filename'], 'value': row['id']} for row in data]


@app.expanded_callback(
    dash.dependencies.Output('image-plot-output-div', 'children'),
    dash.dependencies.Input('file-list-dropdown', 'value'))
def callback_image_plot(*args, **kwargs):
    if args[0] is None:
        return None
    if kwargs['session_state'].get('auth_token') is not None:
        archive_header = {'Authorization': f'Token {kwargs["session_state"]["auth_token"]}'}
    else:
        archive_header = None
    response = requests.get(settings.ARCHIVE_URL + f'/{args[0]}', headers=archive_header)
    response.raise_for_status()
    buffer = io.BytesIO()
    buffer.write(requests.get(response.json()['url'], stream=True).content)
    buffer.seek(0)

    hdu = fits.open(buffer)
    # Create a trace
    trace = go.Heatmap(z=hdu['SCI'].data)

    layout = dict(title='',
                  margin=dict(t=20, b=50, l=50, r=40),
                  height=350)

    fig = dict(data=[trace], layout=layout)
    image_plot = dcc.Graph(id='image-graph1', figure=fig, style={'display': 'inline-block',
                                                                 'width': '100%', 'height': '100%;'})
    children = image_plot

    return [children]
