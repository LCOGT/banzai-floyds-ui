import dash
from dash import dcc, html
from django_plotly_dash import DjangoDash
import logging
import datetime
import requests
import asyncio
from banzai_floyds_ui.gui.utils.file_utils import fetch_all, get_related_frame
from banzai_floyds_ui.gui.plots import make_1d_sci_plot, make_2d_sci_plot, make_arc_2d_plot, make_arc_line_plots
from banzai_floyds_ui.gui.plots import make_profile_plot
from banzai_floyds_ui.gui.utils.plot_utils import extraction_region_traces
from dash.exceptions import PreventUpdate
from banzai_floyds_ui.gui.utils.plot_utils import json_to_polynomial, plot_extracted_data
from banzai_floyds_ui.gui.utils.plot_utils import EXTRACTION_REGION_LINE_ORDER
from banzai.utils import import_utils
from banzai.utils.stage_utils import get_stages_for_individual_frame
from banzai_floyds.frames import FLOYDSObservationFrame
import dash_bootstrap_components as dbc
from banzai_floyds import settings
import os
import banzai.main

logger = logging.getLogger(__name__)

dashboard_name = 'banzai-floyds'
app = DjangoDash(name=dashboard_name)

# set up the context object for banzai

parser.add_argument('--post-to-archive', dest='post_to_archive', action='store_true', default=False)
parser.add_argument('--no-file-cache', dest='no_file_cache', action='store_true', default=False,
                    help='Turn off saving files to disk')
parser.add_argument('--post-to-opensearch', dest='post_to_opensearch', action='store_true',
                    default=False)

settings.fpack=True
settings.db_address = os.environ['DB_ADDRESS']
RUNTIME_CONTEXT = banzai.main.parse_args(settings, parse_system_args=False)


def layout():
    last_week = datetime.date.today() - datetime.timedelta(days=7)

    return html.Div(
        id='main',
        children=[
            html.Div(
                id='options-container',
                children=[
                    html.Div(
                        children=[
                            dcc.DatePickerRange(
                                id='date-range-picker',
                                start_date=last_week.strftime('%Y-%m-%d'),
                                end_date=datetime.date.today().strftime('%Y-%m-%d')
                            ),
                            dcc.Dropdown(
                                id='site-picker',
                                options=[
                                    {'label': 'ogg', 'value': 'en06'},
                                    {'label': 'coj', 'value': 'en12'}
                                ],
                                value=None,
                                placeholder='Site',
                                style={
                                    'float': 'left',
                                    'font-size': '1.1875rem',
                                    'min-width': '7rem',
                                    'height': '3rem'
                                }
                            )
                        ]
                    ),
                    dcc.Loading(
                        id='loading-dropdown',
                        type='default',
                        children=[
                            dcc.Dropdown(
                                id='file-list-dropdown',
                                options=[],
                                placeholder='Select spectrum',
                                value=None,
                                className='col-md-12'
                            )
                        ]
                    )
                ]
            ),
            html.Div(
                id='plot-container',
                children=[
                    dcc.Store(id='initial-extraction-info'),
                    dcc.Store(id='extraction-positions'),
                    dcc.Store(id='extraction-traces'),
                    dcc.Store(id='extractions'),
                    dcc.Loading(
                        id='loading-arc-2d-plot-container',
                        type='default',
                        children=[
                            dcc.Graph(id='arc-2d-plot',
                                      style={'display': 'inline-block',
                                             'width': '100%', 'height': '550px;'}),
                        ]
                    ),
                    dcc.Loading(
                        id='loading-arc-1d-plot-container',
                        type='default',
                        children=[
                            dcc.Graph(id='arc-1d-plot',
                                      style={'display': 'inline-block',
                                             'width': '100%', 'height': '100%;'}),
                        ]
                    ),
                    dcc.Loading(
                        id='loading-sci-2d-plot-container',
                        type='default',
                        children=[
                            dcc.Graph(id='sci-2d-plot',
                                      style={'display': 'inline-block',
                                             'width': '100%', 'height': '550px;'}),
                        ]
                    ),
                    dcc.Loading(
                        id='loading-profile-plot-container',
                        type='default',
                        children=[
                            dcc.Graph(id='profile-plot',
                                      style={'display': 'inline-block',
                                             'width': '100%', 'height': '550px;'},
                                      config={'edits': {'shapePosition': True}}),
                        ]
                    ),
                    html.Div('Extraction Type:'),
                    dcc.Loading(
                        id='loading-extraction-plot-container',
                        type='default',
                        children=[
                            dcc.Graph(id='extraction-plot',
                                      style={'display': 'inline-block',
                                             'width': '100%', 'height': '550px;'}),
                        ]
                    ),
                    dcc.radioItems(['Optimal', 'Unweighted'], 'Optimal', inline=True, id='extraction-type'),
                    dcc.button('Save Extraction', id='extract-button'),
                ]
            )
        ]
    )


app.layout = layout


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
                       'instrument_id': instrument_id, 'RLEVEL': 91, 'basename': '1d'}
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


# This callback snaps the lines back to the same y-position so they don't wander
app.clientside_callback(
    """
    function(extraction_data) {
        if (typeof extraction_data === "undefined") {
            return window.dash_clientside.no_update;
        }

        var dccGraph = document.getElementById('profile-plot');
        var jsFigure = dccGraph.querySelector('.js-plotly-plot');
        var update = {};
        const lines = [
            'center', 'extract_lower', 'extract_upper', 'bkg_left_inner',
            'bkg_right_inner', 'bkg_left_outer', 'bkg_right_outer'
        ];
        const orders = ['2', '1'];
        let index = 0;
        for (const order of orders) {
            for (const line of lines) {
                update[`shapes[${index}].x0`] = extraction_data[order][line];
                update[`shapes[${index}].x1`] = extraction_data[order][line];
                update[`shapes[${index}].y0`] = -2.0;
                update[`shapes[${index}].y1`] = 2.0;
                index++;
            }
        }

        Plotly.relayout(jsFigure, update, []);
        return window.dash_clientside.no_update;
    }
    """,
    dash.dependencies.Input('extraction-positions', 'data'),
    prevent_initial_call=True
)


# This updates the traces on the sci-2d-plot figure without redrawing the whole figure
# which was prohibitively slow
app.clientside_callback(
    """
    function(extraction_traces) {
        if (typeof extraction_traces === "undefined") {
            return window.dash_clientside.no_update;
        }

        var dccGraph = document.getElementById('sci-2d-plot');
        var jsFigure = dccGraph.querySelector('.js-plotly-plot');
        var trace_ids = [];
        for (let i = 1; i <= extraction_traces.x.length; i++) {
            trace_ids.push(i);
        }
        Plotly.restyle(jsFigure, extraction_traces, trace_ids);
        return window.dash_clientside.no_update;
    }
    """,
    dash.dependencies.Input('extraction-traces', 'data'),
    prevet_initial_call=True
)


@app.expanded_callback(dash.dependencies.Output('extraction-traces', 'data'),
                       dash.dependencies.Input('extraction-positions', 'data'),
                       dash.dependencies.State('initial-extraction-info', 'data'),
                       prevent_initial_call=True)
def on_extraction_region_update(extraction_positions, initial_extraction_info):
    xs, ys = [], []
    for order in [2, 1]:
        order_center_polynomial = json_to_polynomial(initial_extraction_info['order_center'][str(order)])
        wavelengths_polynomial = json_to_polynomial(initial_extraction_info['wavelength'][str(order)])
        center_polynomial = json_to_polynomial(initial_extraction_info['profile_center'][str(order)])
        width_polynomal = json_to_polynomial(initial_extraction_info['profile_sigma'][str(order)])
        positions_sigma = {}
        for line in EXTRACTION_REGION_LINE_ORDER[1:]:
            center = extraction_positions[str(order)]['center']
            position = extraction_positions[str(order)][line]
            positions_sigma[line] = abs(position - center)
            positions_sigma[line] /= initial_extraction_info['refsigma'][str(order)]
        center_delta = extraction_positions[str(order)]['center'] - initial_extraction_info['refcenter'][str(order)]
        x, traces = extraction_region_traces(order_center_polynomial, center_polynomial, width_polynomal,
                                             wavelengths_polynomial, **positions_sigma, center_delta=center_delta)

        for trace in traces:
            xs.append(x)
            ys.append(trace)

        return {'x': xs, 'y': ys}


@app.expanded_callback(
    [dash.dependencies.Output('arc-2d-plot', 'figure'),
     dash.dependencies.Output('arc-1d-plot', 'figure'),
     dash.dependencies.Output('sci-2d-plot', 'figure'),
     dash.dependencies.Output('profile-plot', 'figure'),
     dash.dependencies.Output('extraction-plot', 'figure'),
     dash.dependencies.Output('initial-extraction-info', 'data')],
    dash.dependencies.Input('file-list-dropdown', 'value'), prevent_intial_call=True)
def callback_make_plots(*args, **kwargs):
    frame_id = args[0]
    if frame_id is None:
        raise PreventUpdate
    if kwargs['session_state'].get('auth_token') is not None:
        archive_header = {'Authorization': f'Token {kwargs["session_state"]["auth_token"]}'}
    else:
        archive_header = None

    # TODO: All of of these should be async so things load faster
    arc_frame, arc_filename = get_related_frame(frame_id, archive_header, 'L1IDARC')
    arc_image_plot = make_arc_2d_plot(arc_frame, arc_filename)
    arc_line_plot = make_arc_line_plots(arc_frame)

    sci_2d_frame, sci_2d_filename = get_related_frame(frame_id, archive_header, 'L1ID2D')
    sci_2d_plot, extraction_data = make_2d_sci_plot(sci_2d_frame, sci_2d_filename)

    kwargs['session_state'][sci_2d_filename] = sci_2d_frame

    profile_plot, initial_extraction_info = make_profile_plot(sci_2d_frame)

    for key in extraction_data:
        initial_extraction_info[key] = extraction_data[key]

    sci_1d_plot = make_1d_sci_plot(frame_id, archive_header)
    return arc_image_plot, arc_line_plot, sci_2d_plot, profile_plot, sci_1d_plot, initial_extraction_info


@app.expanded_callback(dash.dependencies.Output('extraction-positions', 'data'),
                       [dash.dependencies.Input('initial-extraction-info', 'data'),
                        dash.dependencies.Input('profile-plot', 'relayoutData')],
                       dash.dependencies.State('extraction-positions', 'data'),
                       prevent_initial_call=True)
def update_extraction_positions(initial_extraction_info, relayout_data, current_extraction_positions):
    # Technically we should be able to use ctx.triggered_id here but it doesn't seem to be plumbed through
    # django-plotly-dash
    triggered_id = dash.callback_context.triggered[0]['prop_id'].split('.')[0]

    if triggered_id == 'initial-extraction-info':
        if initial_extraction_info is None:
            raise PreventUpdate
        extraction_positions = {"1": {}, "2": {}}
        for order in [2, 1]:
            for line in EXTRACTION_REGION_LINE_ORDER:
                extraction_positions[str(order)][line] = initial_extraction_info["positions"][str(order)][line]
        return extraction_positions
    # Otherwise we are in the relayout data case
    if current_extraction_positions is None:
        raise PreventUpdate
    key_with_update = None
    for key in relayout_data:
        if 'shapes[' in key and '.x0' in key:
            key_with_update = key
            break
    if key_with_update is None:
        raise PreventUpdate
    line_index = int(key_with_update.split('shapes[')[1].split(']')[0])

    # We need to convert the shape index into the trace index
    # Order matters here. For shapes, we do extraction center, -+ extraction region, background inner edge (left, right),
    # background outer region (left, right), for order 2, 1
    # And a plus 1 for the heatmap.
    # Assuming we did everthing in the same order, you just need to add 1 for the heatmap to get the right trace index
    if line_index < 7:
        order = 2
    else:
        order = 1
    center_line = line_index % 7 == 0
    if center_line:
        position_delta = relayout_data[key_with_update] - current_extraction_positions[str(order)]['center']
        for line in EXTRACTION_REGION_LINE_ORDER:
            current_extraction_positions[str(order)][line] += position_delta
    else:
        line_id = EXTRACTION_REGION_LINE_ORDER[line_index % 7]
        current_extraction_positions[str(order)][line_id] = relayout_data[key_with_update]
    return current_extraction_positions


def reextract(hdu, filename, centers, extraction_positions, runtime_context, extraction_type='optimal'):
    # Convert 2d science hdu to banzai-floyds frame object
    frame = FLOYDSObservationFrame(hdu, )
    # reset the weights and the background region
    _, original_widths = frame.profile_fits
    frame.profile_fits = centers, original_widths, frame['PROFILEFITS'].data
    frame.profile = load_profile_data(frame.profile_fits)
    frame.extraction_window = bar
    if extraction_type == 'unweighted':
        frame.binned_data['weights'] = 1.0
    frame.background_window = foo
    stages_to_do = get_stages_for_individual_frame(runtime_context.ORDERED_STAGES,
                                                   last_stage=runtime_context.LAST_STAGE[frame.obstype.upper()],
                                                   extra_stages=runtime_context.EXTRA_STAGES[frame.obstype.upper()])
    
    # Starting at the extraction weights stage
    start_index = stages_to_do.index('banzai_floyds.extract.Extractor')
    stages_to_do = stages_to_do[start_index:]

    for stage_name in stages_to_do:
        stage_constructor = import_utils.import_attribute(stage_name)
        stage = stage_constructor(runtime_context)
        frames = stage.run([frame])
        if not frames:
            logger.error('Reduction stopped', extra_tags={'filename': filename})
            return
    return frame


@app.expanded_callback(dash.dependencies.Output('extractions', 'data'),
                       [dash.dependencies.Input('extraction-positions', 'data'),
                        dash.dependencies.Input('extraction-type', 'value')], 
                        prevent_initial_call=True)
def trigger_reextract(extraction_positions, extraction_type, **kwargs):

    science_frame = kwargs['session_state'].get('science_2d_frame')
    if science_frame is None:
        raise PreventUpdate

    frame = reextract(science_frame, centers, extraction_positions, 
                      RUNTIME_CONTEXT, extraction_type=extraction_type.lower())
    return plot_extracted_data(frame.extracted)


app.clientside_callback(
    """
    function(extraction_data) {
        if (typeof extraction_data === "undefined") {
            return window.dash_clientside.no_update;
        }

        var dccGraph = document.getElementById('sci-1d-plot');
        var jsFigure = dccGraph.querySelector('.js-plotly-plot');
        var trace_ids = [];
        for (let i = 1; i <= extraction_data.x.length; i++) {
            trace_ids.push(i);
        }
        Plotly.restyle(jsFigure, extraction_data, trace_ids);
        return window.dash_clientside.no_update;
    }
    """,
    dash.dependencies.Input('extractions', 'data'),
    prevent_initial_call=True)


@app.expanded_callback(dash.dependencies.Input('extract-button', 'n_clicks'),
                       [dash.dependencies.State('extraction-type', 'value'),
                        dash.dependencies.State('extraction-positions', 'data')], prevent_initial_call=True)
def save_extraction(n_clicks, extraction_type, extraction_positions, **kwargs):
    if not n_clicks:
        raise PreventUpdate
    science_frame = kwargs['session_state'].get('science_2d_frame')
    if science_frame is None:
        raise PreventUpdate

    # If not logged in, open a modal saying you can only save if you are.
    username = kwargs['session_state'].get('username')
    if username is None:
        throw error

    # Run the reextraction
    extracted_frame = reextract(science_frame, filename, centers, extraction_positions,
                                RUNTIME_CONTEXT, extraction_type=extraction_type.lower())
    # Save the reducer into the header
    extracted_frame.meta['REDUCER'] = username
    # Push the results to the archive
    extracted_frame.write(RUNTIME_CONTEXT)
