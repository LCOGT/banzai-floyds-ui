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
from banzai_floyds_ui.gui.utils.plot_utils import json_to_polynomial
from banzai_floyds_ui.gui.utils.plot_utils import EXTRACTION_REGION_LINE_ORDER


logger = logging.getLogger(__name__)

dashboard_name = 'banzai-floyds'
app = DjangoDash(name=dashboard_name)


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
                    dcc.Loading(
                        id='loading-extraction-plot-container',
                        type='default',
                        children=[
                            dcc.Graph(id='extraction-plot',
                                      style={'display': 'inline-block',
                                             'width': '100%', 'height': '550px;'}),
                        ]
                    )
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
    print(current_extraction_positions)
    if center_line:
        position_delta = relayout_data[key_with_update] - current_extraction_positions[str(order)]['center']
        for line in EXTRACTION_REGION_LINE_ORDER:
            current_extraction_positions[str(order)][line] += position_delta
    else:
        line_id = EXTRACTION_REGION_LINE_ORDER[line_index % 7]
        current_extraction_positions[str(order)][line_id] = relayout_data[key_with_update]
    print(current_extraction_positions)
    return current_extraction_positions
