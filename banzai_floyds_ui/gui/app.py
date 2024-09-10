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
from banzai_floyds.arc_lines import used_lines as arc_lines
from banzai_floyds.orders import orders_from_fits
import numpy as np
from scipy.interpolate import LinearNDInterpolator
from plotly.subplots import make_subplots
from astropy.table import Table
from banzai_floyds_ui.gui.plots import COLORMAP, make_2d_sci_plot, LAVENDER
from banzai_floyds_ui.gui.utils.header_utils import header_to_polynomial


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
            dcc.Loading(
                id='loading-plot-container',
                type='default',
                children=[
                    html.Div(id='arc-plot-output-div')
                ]
            )
        ]
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


def make_arc_2d_plot(arc_frame_hdu, arc_filename):
    zmin, zmax = np.percentile(arc_frame_hdu['SCI'].data, [1, 99])
    trace = go.Heatmap(z=arc_frame_hdu['SCI'].data, colorscale=COLORMAP, zmin=zmin, zmax=zmax, hoverinfo='none',
                       colorbar=dict(title='Data (counts)'))

    layout = dict(margin=dict(t=50, b=50, l=50, r=40), height=370)

    orders = orders_from_fits(arc_frame_hdu['ORDER_COEFFS'].data, arc_frame_hdu['ORDER_COEFFS'].header,
                              arc_frame_hdu['SCI'].data.shape)
    x2d, y2d = np.meshgrid(np.arange(orders.shape[1]), np.arange(orders.shape[0]))
    figure_data = [trace]

    # Make lines at each of the atlas arc lines for each order
    for order, order_height in zip(orders.order_ids, orders.order_heights):
        in_order = order == orders.data
        wavelength_to_x_interpolator = LinearNDInterpolator((y2d[in_order].ravel(),
                                                             arc_frame_hdu['WAVELENGTHS'].data[in_order]),
                                                            x2d[in_order].ravel())
        xmin, xmax = np.min(x2d[in_order]), np.max(x2d[in_order])
        x = np.arange(xmin, xmax + 1)
        y_center = orders.center(x)[order - 1]
        x_y_to_wavelength_intepolator = LinearNDInterpolator((x2d[in_order], y2d[in_order]),
                                                             arc_frame_hdu['WAVELENGTHS'].data[in_order])
        min_wavelength = np.min(arc_frame_hdu['WAVELENGTHS'].data[in_order])
        max_wavelength = np.max(arc_frame_hdu['WAVELENGTHS'].data[in_order])
        for i, line in enumerate(arc_lines):
            if min_wavelength <= line['wavelength'] <= max_wavelength:
                y_line_center = np.interp(line['wavelength'], x_y_to_wavelength_intepolator(x, y_center), y_center)
                y_line = np.arange(y_line_center - order_height / 2.0, y_line_center + order_height / 2.0 + 1)
                plot_x = wavelength_to_x_interpolator(y_line, np.ones_like(y_line) * line['wavelength'])
                if i == 0:
                    show_legend = True
                    name = 'Model'
                else:
                    show_legend = False
                    name = None
                figure_data.append(go.Scatter(x=plot_x, y=y_line,
                                              marker={'color': 'salmon'},
                                              mode='lines',
                                              hovertext=[f'{line["wavelength"]:0.3f} {line["line_source"]}'
                                                         for _ in range(len(plot_x))],
                                              hovertemplate='%{hovertext}<extra></extra>',
                                              showlegend=show_legend,
                                              name=name))
    layout['legend'] = dict(x=0, y=0.95)
    layout['title'] = f'Arc Frame Used in Reduction: {arc_filename}'
    layout['xaxis'] = dict(title='x (pixel)')
    layout['yaxis'] = dict(title='y (pixel)')
    fig = dict(data=figure_data, layout=layout)
    image_plot = dcc.Graph(id='image-graph1', figure=fig, style={'display': 'inline-block',
                                                                 'width': '100%', 'height': '100%;'},)
    return image_plot


def download_frame(headers, url=f'{settings.ARCHIVE_URL}', params=None, list_endpoint=False):
    response = requests.get(url, params=params, headers=headers)
    response.raise_for_status()

    buffer = io.BytesIO()
    if list_endpoint:
        buffer.write(requests.get(response.json()['results'][0]['url'], stream=True).content)
    else:
        buffer.write(requests.get(response.json()['url'], stream=True).content)
    buffer.seek(0)

    hdu = fits.open(buffer)
    return hdu


def get_related_frame(frame_id, archive_header, related_frame_key):
    # Get the related frame from the archive that matches related_frame_key in the header.
    response = requests.get(f'{settings.ARCHIVE_URL}{frame_id}/headers', headers=archive_header)
    response.raise_for_status()
    related_frame_filename = response.json()['data'][related_frame_key]
    params = {'basename_exact': related_frame_filename}
    return download_frame(archive_header, params=params, list_endpoint=True), related_frame_filename


def make_arc_line_plots(arc_frame_hdu):
    """Make a plot for each order showing the arc lines and their residulas"""

    fig = make_subplots(rows=2, cols=2, vertical_spacing=0.02,
                        horizontal_spacing=0.05, shared_xaxes=True)

    fig.update_yaxes(title_text='Flux (counts)', row=1, col=1)
    fig.update_yaxes(title_text='Residuals (\u212B)', row=2, col=1)
    fig.update_xaxes(title_text='Wavelength (\u212B)', row=2, col=1, tickformat=".0f")
    fig.update_xaxes(title_text='Wavelength (\u212B)', row=2, col=2, tickformat=".0f")
    fig.add_annotation(xref='x domain', yref='y domain', x=0.01, y=0.97, text='Blue Order (order=2)', showarrow=False)
    fig.add_annotation(xref='x2 domain', yref='y2 domain', x=0.01, y=0.97, text='Red Order (order=1)', showarrow=False)

    extracted_data = arc_frame_hdu['EXTRACTED'].data
    lines_used = arc_frame_hdu['LINESUSED'].data
    plot_column = {2: 1, 1: 2}
    for order in [2, 1]:
        where_order = extracted_data['order'] == order
        fig.add_trace(
            go.Scatter(x=extracted_data['wavelength'][where_order], y=extracted_data['fluxraw'][where_order],
                       line_color='#023858', mode='lines'),
            row=1, col=plot_column[order],
        )
        flux = extracted_data['fluxraw'][where_order]
        wavelength = extracted_data['wavelength'][where_order]
        flux_range = np.max(flux) - np.min(flux)
        # Length of the annotation tick
        annotation_height = 0.1 * flux_range
        # offset from the flux for the tick
        annotation_sep = 0.05 * flux_range

        line_fluxes = np.interp([line['wavelength'] for line in arc_lines], wavelength, flux)
        # Add tick marks above the expected lines
        for line, line_flux in zip(arc_lines, line_fluxes):
            # The number of points in the tick line is arbitrary. You just want enough points that the
            # hover text is easy to see as plotly can't do hover text over regions yet.
            if not np.min(wavelength) <= line['wavelength'] <= np.max(wavelength):
                continue
            annotation_y = np.linspace(line_flux + annotation_sep, line_flux + annotation_sep + annotation_height, 11)
            annotation_x = np.ones_like(annotation_y) * line['wavelength']
            annotation_hovertext = [f'{line["wavelength"]:0.3f} {line["line_source"]}' for _ in annotation_x]
            fig.add_trace(go.Scatter(x=annotation_x, y=annotation_y, hovertext=annotation_hovertext,
                                     mode='lines',  marker={'color': 'salmon'},
                                     hovertemplate='%{hovertext}<extra></extra>'),
                          row=1, col=plot_column[order])
        lines_used = arc_frame_hdu['LINESUSED'].data[arc_frame_hdu['LINESUSED'].data['order'] == order]
        reference_wavelengths = lines_used['reference_wavelength']

        residual_hover_text = [f'{line["wavelength"]:0.3f} {line["line_source"]}' for line in arc_lines
                               if line['wavelength'] in reference_wavelengths]
        residuals_wavelengths = lines_used['measured_wavelength']
        residuals = residuals_wavelengths - reference_wavelengths
        fig.add_trace(
            go.Scatter(x=residuals_wavelengths, y=residuals,
                       mode='markers', marker=dict(color='#023858'),
                       hovertext=residual_hover_text,
                       hovertemplate='%{y}\u212B: %{hovertext}<extra></extra>'),
            row=2, col=plot_column[order],
        )
        residual_range = np.max(residuals) - np.min(residuals)

        fig.update_yaxes(range=[np.min(residuals) - 0.1 * residual_range,
                                np.max(residual_range) + 0.1 * residual_range],
                         row=2, col=plot_column[order])
    #    fig.add_annotation(xref=, yref=, x=, y=, text='', showarrow=False)
    fig.update_layout(showlegend=False, autosize=True, margin=dict(l=0, r=0, t=0, b=0))
    line_plot = dcc.Graph(id='arc-line-graph', figure=fig, style={'display': 'inline-block',
                                                                  'width': '100%', 'height': '550px;'})
    return line_plot


def unfilled_histogram(x, y, color, name=None, legend=None):
    # I didn't like how the plotly histogram looked so I wrote my own
    x_avgs = (x[1:] + x[:-1]) / 2.0
    x_lows = np.hstack([x[0] + x[0] - x_avgs[0], x_avgs])
    x_highs = np.hstack([x_avgs, x[-1] + x[-1] - x_avgs[-1]])
    x_plot = []
    y_plot = []
    for x_low, x_center, x_high, y_center in zip(x_lows, x, x_highs, y):
        x_plot.append(x_low)
        x_plot.append(x_center)
        x_plot.append(x_high)
        # Make the flat top at -1, 0, +1 of x
        for _ in range(3):
            y_plot.append(y_center / np.max(y))
    show_legend = name is not None
    return go.Scatter(x=x_plot, y=y_plot, mode='lines', line={'color': color}, hoverinfo='skip',
                      name=name, showlegend=show_legend, legend=legend)


def make_profile_plot(sci_2d_frame):
    layout = dict(title='', margin=dict(t=20, b=20, l=50, r=40), height=720, showlegend=True, shapes=[])
    fig = make_subplots(rows=2, cols=2, vertical_spacing=0.13,
                        subplot_titles=("Profile Cross Section: Blue Order (order=2)",
                                        "Profile Center: Blue Order (order=2)",
                                        "Profile Cross Section: Red Order (order=1)",
                                        "Profile Center: Red Order (order=1)"))
    plot_row = {2: 1, 1: 2}
    # Define the coordinate refernce plot manually per order
    reference_axes = {2: 1, 1: 3}
    # Approximate wavelength center to plot the profile
    order_center = {1: 7000, 2: 4500}
    for order in [2, 1]:
        binned_data = Table(sci_2d_frame['BINNED2D'].data).group_by(('order', 'wavelength_bin'))
        # We have to remove the last index here because astropy prepolulates it with the final row in the table so it
        # knows where to start if you add a new group
        wavelength_bins = np.array([binned_data[index]['wavelength_bin'] for index in binned_data.groups.indices[:-1]])
        closest_wavelength_bin = np.argmin(np.abs(wavelength_bins - order_center[order]))
        data = binned_data[binned_data.groups.indices[closest_wavelength_bin]:
                           binned_data.groups.indices[closest_wavelength_bin + 1]]

        if order == 2:
            model_name = 'Model'
            data_name = 'Data'
        else:
            model_name = None
            data_name = None
        fig.add_trace(
            unfilled_histogram(data['y_order'], data['data'], '#023858', name=data_name, legend='legend'),
            row=plot_row[order], col=1
        )

        fig.add_trace(
            unfilled_histogram(data['y_order'], data['weights'], 'salmon', name=model_name, legend='legend'),
            row=plot_row[order], col=1
        )

        traced_points = sci_2d_frame['PROFILEFITS'].data[sci_2d_frame['PROFILEFITS'].data['order'] == order]
        fig.add_trace(
            go.Scatter(x=traced_points['wavelength'],
                       y=traced_points['center'],
                       mode='markers', marker={'color': '#023858'},
                       hoverinfo='skip', showlegend=True, legend='legend2'),
            row=plot_row[order], col=2,
            )
        center_polynomial = header_to_polynomial(sci_2d_frame['PROFILEFITS'].header, 'CTR', order)
        x_plot = np.arange(center_polynomial.domain[0], center_polynomial.domain[1] + 1, 1.0)
        fig.add_trace(
            go.Scatter(x=x_plot, y=center_polynomial(x_plot), mode='lines', line={'color': 'salmon'},
                       hoverinfo='skip', showlegend=True, legend='legend2'),
            row=plot_row[order], col=2,
            )
        # Add in the extraction center and region and background region lines
        # We do this based on header keywords, but we really should do it on the binned data
        extract_center = center_polynomial(order_center[order])
        width_polynomial = header_to_polynomial(sci_2d_frame['PROFILEFITS'].header, 'WID', order)
        extract_sigma = width_polynomial(order_center[order])
        n_extract_sigma = sci_2d_frame['SCI'].header['EXTRTWIN']
        bkg_lower_n_sigma = sci_2d_frame['SCI'].header['BKWINDW0']
        bkg_upper_n_sigma = sci_2d_frame['SCI'].header['BKWINDW1']
        layout['shapes'].append({'type': 'line',
                                 'x0': extract_center, 'x1': extract_center,
                                 'y0': -0.2, 'y1': 1.2,
                                 'name': 'Extraction Center',
                                 'line': {'color': LAVENDER, 'width': 2},
                                 'xref': f'x{reference_axes[order]}', 'yref': f'y{reference_axes[order]}'})
        # Add dummy traces to make the legend... ugh...
        fig.add_trace(go.Scatter(x=[None], y=[None], mode='lines',
                                 line={'color': LAVENDER, 'width': 2}, name='Extraction Center', legend='legend',
                                 showlegend=True), row=plot_row[order], col=1)

    fig.update_yaxes(title_text='Normalized Flux', range=[-0.1, 1.1], row=1, col=1)
    fig.update_xaxes(title_text='y offset from center (pixel)', row=1, col=1)
    fig.update_yaxes(title_text='Normalized Flux', range=[-0.1, 1.1], row=2, col=1)
    fig.update_xaxes(title_text='y offset from center (pixel)', row=2, col=1)

    fig.update_yaxes(title_text='y (pixel)', row=1, col=2)
    fig.update_xaxes(title_text='x (pixel)', row=1, col=2)
    fig.update_yaxes(title_text='y (pixel)', row=2, col=2)
    fig.update_xaxes(title_text='x (pixel)', row=2, col=2)

    # Split the legends for each subplot
    layout['legend'] = dict(yref="container", xref='container', y=0.99, x=0.01)
    layout['legend2'] = dict(yref="container", xref='container', y=0.3, x=0.99)

    fig.update_layout(**layout)
    profile_plot = dcc.Graph(id='profile-graph', figure=fig,
                             style={'display': 'inline-block', 'width': '100%', 'height': '100%;'},
                             config={'editable': True, 'edits': {'shapePosition': True}})
    return profile_plot


def make_1d_sci_plot(frame_id, archive_header):

    frame_1d = download_frame(url=f'{settings.ARCHIVE_URL}{frame_id}/', headers=archive_header)
    frame_data = frame_1d[1].data
    title_dict = {
        'text': f"1-D Extractions: {frame_1d[0].header['ORIGNAME'].replace('-e00', '-e91-1d')}",
        'y': 0.985,
        'x': 0.5,
        'xanchor': 'center',
        'yanchor': 'top'
        }
    layout = dict(title=title_dict, margin=dict(t=60, b=50, l=0, r=0), height=1080, showlegend=False)
    fig = make_subplots(rows=3, cols=2, vertical_spacing=0.02, horizontal_spacing=0.07,
                        shared_xaxes=True, subplot_titles=['Blue Order (order=2)', 'Red Order (order=1)',
                                                           None, None, None, None])
    plot_column = {2: 1, 1: 2}
    for order in [2, 1]:
        where_order = frame_data['order'] == order
        fig.add_trace(
            go.Scatter(x=frame_data['wavelength'][where_order], y=frame_data['flux'][where_order],
                       line_color='#023858', mode='lines'),
            row=1, col=plot_column[order],
        )
        fig.add_trace(
            go.Scatter(x=frame_data['wavelength'][where_order], y=frame_data['fluxraw'][where_order],
                       line_color='#023858', mode='lines'),
            row=2, col=plot_column[order],
        )
        fig.add_trace(
            go.Scatter(x=frame_data['wavelength'][where_order], y=frame_data['background'][where_order],
                       line_color='#023858', mode='lines'),
            row=3, col=plot_column[order],
        )
    fig.update_yaxes(title_text='Flux (erg s\u207B\u00B9 cm\u207B\u00B2 \u212B\u207B\u00B9)', row=1, col=1,
                     exponentformat='power')
    fig.update_yaxes(row=1, col=2, exponentformat='power')
    fig.update_yaxes(title_text='Flux (counts)', row=2, col=1, exponentformat='power')
    fig.update_yaxes(row=2, col=2, exponentformat='power')
    fig.update_yaxes(title_text='Background (counts)', row=3, col=1, exponentformat='power')
    fig.update_yaxes(row=3, col=2, exponentformat='power')
    fig.update_xaxes(title_text='Wavelength (\u212B)', row=3, col=1, tickformat=".0f")
    fig.update_xaxes(title_text='Wavelength (\u212B)', row=3, col=2, tickformat=".0f")
    fig.update_layout(**layout)

    extracted_plot = dcc.Graph(id='extracted-graph', figure=fig,
                               style={'display': 'inline-block', 'width': '100%', 'height': '100%;'})
    return extracted_plot


@app.expanded_callback(
    dash.dependencies.Output('arc-plot-output-div', 'children'),
    dash.dependencies.Input('file-list-dropdown', 'value'))
def callback_make_plots(*args, **kwargs):
    frame_id = args[0]
    if frame_id is None:
        return None
    if kwargs['session_state'].get('auth_token') is not None:
        archive_header = {'Authorization': f'Token {kwargs["session_state"]["auth_token"]}'}
    else:
        archive_header = None

    # TODO: All of of these should be async so things load faster
    arc_frame, arc_filename = get_related_frame(frame_id, archive_header, 'L1IDARC')
    arc_image_plot = make_arc_2d_plot(arc_frame, arc_filename)
    arc_line_plot = make_arc_line_plots(arc_frame)

    sci_2d_frame, sci_2d_filename = get_related_frame(frame_id, archive_header, 'L1ID2D')
    sci_2d_plot = make_2d_sci_plot(sci_2d_frame, sci_2d_filename)

    profile_plot = make_profile_plot(sci_2d_frame)
    sci_1d_plot = make_1d_sci_plot(frame_id, archive_header)
    # Return the plot as the child of the output container div
    return [arc_image_plot, arc_line_plot, sci_2d_plot, profile_plot, sci_1d_plot]
