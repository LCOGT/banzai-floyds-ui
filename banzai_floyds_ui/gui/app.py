import dash
from dash import dcc, html, Output
import dash_bootstrap_components as dbc
from django_plotly_dash import DjangoDash
import datetime
import requests
import asyncio
from banzai_floyds_ui.gui.utils.file_utils import download_frame
from banzai_floyds_ui.gui.utils.file_utils import fetch_all, get_related_frame
from banzai_floyds_ui.gui.plots import make_1d_sci_plot, make_2d_sci_plot, make_arc_2d_plot, make_arc_line_plots
from banzai_floyds_ui.gui.plots import make_profile_plot, make_combined_extraction_plot
from banzai_floyds_ui.gui.utils import file_utils
from banzai_floyds_ui.gui.utils.plot_utils import extraction_region_traces
from dash.exceptions import PreventUpdate
from banzai_floyds_ui.gui.utils.plot_utils import json_to_polynomial
from banzai_floyds_ui.gui.utils.plot_utils import EXTRACTION_REGION_LINE_ORDER
from banzai.utils import import_utils
from banzai.utils.stage_utils import get_stages_for_individual_frame
from banzai_floyds.frames import FLOYDSFrameFactory
from banzai_floyds import settings
from django.conf import settings as django_settings
import os
import banzai.main
import io
from banzai.logs import get_logger
from django.core.cache import cache


logger = get_logger()

dashboard_name = 'banzai-floyds'
app = DjangoDash(name=dashboard_name)

# set up the context object for banzai

settings.fpack = True
settings.post_to_open_search = bool(os.environ.get('POST_TO_OPENSEARCH', False))
settings.post_to_archive = bool(os.environ.get('POST_TO_ARCHIVE', False))
settings.no_file_cache = True
settings.db_address = os.environ['BANZAI_DB_ADDRESS']
RUNTIME_CONTEXT = banzai.main.parse_args(settings, parse_system_args=False)


def layout():
    last_week = datetime.date.today() - datetime.timedelta(days=7)
    docs_link = 'https://banzai-floyds.readthedocs.io/en/latest/banzai_floyds/processing.html'
    optimal_extraction_help_text = 'Performs an extraction based on the algorithm and\n' \
                                   'methodology described by Horne 1986, PASP, 98, 609.'
    unweighted_extraction_help_text = 'Performs an unweighted extraction.'

    return html.Div(
        id='main',
        children=[
            html.Div(
                id='options-container',
                children=[
                    dbc.Modal([
                            dbc.ModalHeader(dbc.ModalTitle("Error"), className='bg-danger text-white'),
                            dbc.ModalBody("You must be logged in to save an extraction."),
                        ],
                        id="error-logged-in-modal",
                        is_open=False,
                    ),
                    dbc.Modal([
                            dbc.ModalHeader(dbc.ModalTitle("Error"), className='bg-danger text-white'),
                            dbc.ModalBody("Error extracting spectrum. Plots may not reflect extraction paramters."),
                        ],
                        id="error-extract-failed-modal",
                        is_open=False,
                    ),
                    dbc.Modal([
                            dbc.ModalHeader(dbc.ModalTitle("Error"), className='bg-danger text-white'),
                            dbc.ModalBody("""
                                          Error saving spectrum.
                                          Note you need to have clicked the re-extract button at least before saving.
                                          """),
                        ],
                        id="error-extract-failed-on-save-modal",
                        is_open=False,
                    ),
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
                    dcc.Store(id='file-list-metadata'),
                    dcc.Store(id='extraction-positions'),
                    dcc.Store(id='extraction-traces'),
                    dcc.Store(id='extractions'),
                    dcc.Store(id='combined-extraction'),
                    html.Div([
                        html.H3(['Wavelength Calibration:',
                                 html.A('?',
                                        href=docs_link+'#wavelength-solution',
                                        className='help-mark',
                                        target="_blank",
                                        title='Click for Docs')
                                 ],),
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
                    ], className='plot-group'),
                    html.Div([
                        html.H3(['Profile Fit:',
                                 html.A('?',
                                        href=docs_link+'#profile-fitting',
                                        className='help-mark',
                                        target="_blank",
                                        title='Click for Docs'),
                                 ]),
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
                    ], className='plot-group'),
                    html.Div([html.Span(['Extraction Type:',
                                         html.A('?',
                                                href=docs_link+'#extraction',
                                                className='help-mark',
                                                target="_blank",
                                                title='Click for Docs')],
                                        ),
                             dcc.RadioItems([{"label": html.Span(['Optimal', html.Span('?', className='help-mark')],
                                                                 title=optimal_extraction_help_text,
                                                                 style={"margin-right": "10px"}),
                                              "value":'Optimal'},
                                             {"label": html.Span(['Unweighted', html.Span('?', className='help-mark')],
                                                                 title=unweighted_extraction_help_text),
                                              "value":'Unweighted'}],
                                            'Optimal', inline=True, id='extraction-type',
                                            ),
                             dbc.Button('Re-Extract', id='extract-button')]),
                    html.Div([
                        html.H3(['Extractions:',
                                 html.A('?',
                                        href=docs_link+'#extraction',
                                        className='help-mark',
                                        target="_blank",
                                        title='Click for Docs')
                                 ],),
                        dcc.Loading(
                            id='loading-extraction-plot-container',
                            type='default',
                            children=[
                                html.Div([
                                    dcc.Graph(id='extraction-plot',
                                              style={'display': 'inline-block',
                                                     'width': '100%', 'height': '550px;'}),
                                    dcc.Graph(id='combined-extraction-plot',
                                              style={'display': 'inline-block',
                                                     'width': '100%', 'height': '350px;'})
                                ], id='extraction-plot-container')
                            ]
                        ),
                    ], className='plot-group'),
                    dbc.Button(html.Span(['Save Extraction',
                                          html.Span('?', className='help-mark')
                                          ]), id='save-button', title="Overwrite existing extraction in Archive"),
                ]
            )
        ]
    )


app.layout = layout


@app.expanded_callback([dash.dependencies.Output('file-list-dropdown', 'options'),
                        dash.dependencies.Output('file-list-metadata', 'data')],
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
        results += [{'label': f'{row["filename"]} {row["OBJECT"]} {row["PROPID"]}', 'value': row['id']} for row in data]
    results.sort(key=lambda x: x['label'])
    return results, results


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
    prevent_initial_call=True
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
        width_polynomial = json_to_polynomial(initial_extraction_info['profile_sigma'][str(order)])
        positions_sigma = {}
        for line in EXTRACTION_REGION_LINE_ORDER[1:]:
            center = extraction_positions[str(order)]['center']
            position = extraction_positions[str(order)][line]
            positions_sigma[line] = position - center
            positions_sigma[line] /= initial_extraction_info['refsigma'][str(order)]
        center_delta = extraction_positions[str(order)]['center'] - initial_extraction_info['refcenter'][str(order)]
        x, traces = extraction_region_traces(order_center_polynomial, center_polynomial, width_polynomial,
                                             wavelengths_polynomial, **positions_sigma, center_delta=center_delta)

        for trace in traces:
            xs.append(x)
            ys.append(trace)

    return {'x': xs, 'y': ys}


@app.expanded_callback(
    [Output('arc-2d-plot', 'figure'),
     Output('arc-1d-plot', 'figure'),
     Output('sci-2d-plot', 'figure'),
     Output('profile-plot', 'figure'),
     Output('extraction-plot', 'figure'),
     Output('combined-extraction-plot', 'figure'),
     Output('initial-extraction-info', 'data')],
    dash.dependencies.Input('file-list-dropdown', 'value'), prevent_initial_call=True)
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

    file_utils.cache_fits('science_2d_frame', sci_2d_frame)
    cache.set('filename', sci_2d_frame['SCI'].header['ORIGNAME'])

    profile_plot, initial_extraction_info = make_profile_plot(sci_2d_frame)

    for key in extraction_data:
        initial_extraction_info[key] = extraction_data[key]

    frame_1d = download_frame(url=f'{django_settings.ARCHIVE_URL}{frame_id}/', headers=archive_header)

    sci_1d_plot = make_1d_sci_plot(frame_1d)
    combined_sci_plot = make_combined_extraction_plot(frame_1d)

    return arc_image_plot, arc_line_plot, sci_2d_plot, profile_plot, sci_1d_plot, \
        combined_sci_plot, initial_extraction_info


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


def reextract(hdu, filename, extraction_positions, initial_extraction_info, runtime_context, extraction_type='optimal'):
    # Convert 2d science hdu to banzai-floyds frame object
    factory = FLOYDSFrameFactory()
    buffer = io.BytesIO()
    hdu.writeto(buffer)
    buffer.seek(0)
    file_info = {'filename': filename, 'data_buffer': buffer}
    frame = factory.open(file_info, runtime_context)
    # reset the weights and the background region
    centers, widths = frame.profile_fits
    for order in [1, 2]:
        center_delta = extraction_positions[str(order)]['center'] - \
            initial_extraction_info['positions'][str(order)]['center']
        centers[order-1].coef[0] += center_delta
    frame.profile = centers, widths, frame['PROFILEFITS'].data

    extraction_windows = []
    for order in [1, 2]:
        lower = extraction_positions[str(order)]['extract_lower'] - extraction_positions[str(order)]['center']
        lower /= initial_extraction_info['refsigma'][str(order)]
        upper = extraction_positions[str(order)]['extract_upper'] - extraction_positions[str(order)]['center']
        upper /= initial_extraction_info['refsigma'][str(order)]
        extraction_windows.append([lower, upper])

    frame.extraction_windows = extraction_windows
    # Override the default optimal extraction weights from the profile
    if extraction_type == 'unweighted':
        frame.binned_data['weights'] = 1.0

    background_windows = []
    for order in [1, 2]:
        order_background = []
        for region in ['left', 'right']:
            inner = extraction_positions[str(order)][f'bkg_{region}_inner'] - extraction_positions[str(order)]['center']
            inner /= initial_extraction_info['refsigma'][str(order)]
            outer = extraction_positions[str(order)][f'bkg_{region}_outer'] - extraction_positions[str(order)]['center']
            outer /= initial_extraction_info['refsigma'][str(order)]
            this_background = [inner, outer]
            this_background.sort()
            order_background.append(this_background)
        background_windows.append(order_background)
    frame.background_windows = background_windows
    stages_to_do = get_stages_for_individual_frame(runtime_context.ORDERED_STAGES,
                                                   last_stage=runtime_context.LAST_STAGE[frame.obstype.upper()],
                                                   extra_stages=runtime_context.EXTRA_STAGES[frame.obstype.upper()])

    # Starting at the extraction weights stage
    start_index = stages_to_do.index('banzai_floyds.background.BackgroundFitter')
    stages_to_do = stages_to_do[start_index:]
    frames = [frame]
    for stage_name in stages_to_do:
        stage_constructor = import_utils.import_attribute(stage_name)
        stage = stage_constructor(runtime_context)
        frames = stage.run(frames)
        if not frames:
            logger.error('Reduction stopped', extra_tags={'filename': filename})
            return
    logger.info('Reduction complete', extra_tags={'filename': filename})
    return frames[0]


# Note we include a container here for the extraction plots that always returns no update
# We have to do this to get the loading spinner to trigger during the re-extraction
@app.expanded_callback([dash.dependencies.Output('extractions', 'data'),
                        dash.dependencies.Output('combined-extraction', 'data'),
                        dash.dependencies.Output('error-extract-failed-modal', 'is_open'),
                        dash.dependencies.Output('extraction-plot-container', 'children')],
                       dash.dependencies.Input('extract-button', 'n_clicks'),
                       [dash.dependencies.State('extraction-positions', 'data'),
                        dash.dependencies.State('extraction-type', 'value'),
                       dash.dependencies.State('initial-extraction-info', 'data')],
                       prevent_initial_call=True)
def trigger_reextract(n_clicks, extraction_positions, extraction_type, initial_extraction_info):
    if not n_clicks:
        raise PreventUpdate
    science_frame = file_utils.get_cached_fits('science_2d_frame')
    if science_frame is None:
        raise PreventUpdate
    filename = cache.get('filename')
    frame = reextract(science_frame, filename, extraction_positions, initial_extraction_info,
                      RUNTIME_CONTEXT, extraction_type=extraction_type.lower())

    if frame is None:
        return dash.no_update, dash.no_update, True, dash.no_update

    file_utils.cache_frame('reextracted_frame', frame)
    x = []
    y = []
    for order in [2, 1]:
        where_order = frame.extracted['order'] == order
        for flux in ['flux', 'fluxraw', 'background']:
            x.append(frame.extracted['wavelength'][where_order])
            y.append(frame.extracted[flux][where_order])
    return {'x': x, 'y': y}, {'x': frame.spectrum['wavelength'], 'y': frame.spectrum['flux']}, False, dash.no_update


app.clientside_callback(
    """
    function(extraction_data) {
        if (typeof extraction_data === "undefined") {
            return window.dash_clientside.no_update;
        }
        var dccGraph = document.getElementById('extraction-plot');
        var jsFigure = dccGraph.querySelector('.js-plotly-plot');
        var trace_ids = [];
        for (let i = 0; i < extraction_data.x.length; i++) {
            trace_ids.push(i);
        }
        Plotly.restyle(jsFigure, extraction_data, trace_ids);
        return window.dash_clientside.no_update;
    }
    """,
    dash.dependencies.Input('extractions', 'data'),
    prevent_initial_call=True)


app.clientside_callback(
    """
    function(combined_extraction_data) {
        if (typeof extraction_data === "undefined") {
            return window.dash_clientside.no_update;
        }
        var dccGraph = document.getElementById('combined-extraction-plot');
        var jsFigure = dccGraph.querySelector('.js-plotly-plot');

        Plotly.restyle(jsFigure, combined_extraction_data, 0);
        return window.dash_clientside.no_update;
    }
    """,
    dash.dependencies.Input('combined-extraction', 'data'),
    prevent_initial_call=True)


@app.expanded_callback([dash.dependencies.Output("error-logged-in-modal", "is_open"),
                        dash.dependencies.Output('error-extract-failed-on-save-modal', 'is_open')],
                       dash.dependencies.Input('save-button', 'n_clicks'),
                       prevent_initial_call=True)
def save_extraction(n_clicks, **kwargs):
    if not n_clicks:
        raise PreventUpdate

    # If not logged in, open a modal saying you can only save if you are.
    username = kwargs['session_state'].get('username')
    if username is None:
        return True, dash.no_update

    # Run the reextraction
    extracted_frame = file_utils.get_cached_frame('reextracted_frame')
    if extracted_frame is None:
        return dash.no_update, True

    # Save the reducer into the header
    extracted_frame.meta['REDUCER'] = username
    # Push the results to the archive
    extracted_frame.write(RUNTIME_CONTEXT)
    # Return false to keep the error modal closed
    return dash.no_update, dash.no_update
