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
from banzai_floyds.wavelengths import WavelengthSolution, identify_peaks, refine_peak_centers
from banzai_floyds.extract import get_wavelength_bins, extract, bin_data
import numpy as np
from scipy.interpolate import LinearNDInterpolator
from plotly.subplots import make_subplots
from banzai_floyds.utils.fitting_utils import gauss
from banzai_floyds.matched_filter import optimize_match_filter
from banzai_floyds.extract import profile_gauss_fixed_width
from scipy.optimize import curve_fit


logger = logging.getLogger(__name__)

dashboard_name = 'banzai-floyds'
app = DjangoDash(name=dashboard_name)

COLORMAP = [
    [0, '#fff7fb'],
    [0.125, '#ece7f2'],
    [0.25, '#d0d1e6'],
    [0.375, '#a6bddb'],
    [0.5, '#74a9cf'],
    [0.625, '#3690c0'],
    [0.75, '#0570b0'],
    [0.875, '#045a8d'],
    [1, '#023858']
]


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


def make_arc_2d_plot(arc_frame_hdu):
    zmin, zmax = np.percentile(arc_frame_hdu['SCI'].data, [1, 99])
    trace = go.Heatmap(z=arc_frame_hdu['SCI'].data, colorscale=COLORMAP, zmin=zmin, zmax=zmax, hoverinfo='none')

    layout = dict(title='', margin=dict(t=20, b=50, l=50, r=40), height=350)

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
        for line in arc_lines:
            if min_wavelength <= line['wavelength'] <= max_wavelength:
                y_line_center = np.interp(line['wavelength'], x_y_to_wavelength_intepolator(x, y_center), y_center)
                y_line = np.arange(y_line_center - order_height / 2.0, y_line_center + order_height / 2.0 + 1)
                plot_x = wavelength_to_x_interpolator(y_line, np.ones_like(y_line) * line['wavelength'])
                figure_data.append(go.Scatter(x=plot_x, y=y_line,
                                              marker={'color': 'salmon'},
                                              mode='lines',
                                              hovertext=[f'{line["wavelength"]:0.3f} {line["line_source"]}'
                                                         for _ in range(len(plot_x))],
                                              hovertemplate='%{hovertext}<extra></extra>',
                                              showlegend=False))
    fig = dict(data=figure_data, layout=layout)
    image_plot = dcc.Graph(id='image-graph1', figure=fig, style={'display': 'inline-block',
                                                                 'width': '100%', 'height': '100%;'})
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
    # Get the related arc frame from the archive
    response = requests.get(f'{settings.ARCHIVE_URL}{frame_id}/headers', headers=archive_header)
    response.raise_for_status()
    related_frame_filename = response.json()['data'][related_frame_key]
    params = {'basename_exact': related_frame_filename}
    return download_frame(archive_header, params=params, list_endpoint=True)


def calculate_residuals(wavelengths, flux, flux_errors, lines):
    residuals = []
    residuals_wavelengths = []
    peaks = np.array(identify_peaks(flux, flux_errors, 4, 10, snr_threshold=15.0))
    for line in lines:
        if line['wavelength'] > np.max(wavelengths) or line['wavelength'] < np.min(wavelengths):
            continue
        closest_peak = peaks[np.argmin(np.abs(wavelengths[peaks] - line['wavelength']))]
        closest_peak_wavelength = wavelengths[closest_peak]
        if np.abs(closest_peak_wavelength - line['wavelength']) <= 20:
            refined_peak = refine_peak_centers(flux, flux_errors, np.array([closest_peak]), 4)[0]
            if not np.isfinite(refined_peak):
                continue
            if np.abs(refined_peak - closest_peak) > 5:
                continue
            refined_peak = np.interp(refined_peak, np.arange(len(wavelengths)), wavelengths)
            residuals.append(refined_peak - line['wavelength'])
            residuals_wavelengths.append(line['wavelength'])
    return np.array(residuals_wavelengths), np.array(residuals)


def make_arc_line_plots(arc_frame_hdu):
    """Make a plot for each order showing the arc lines and their residulas"""
    # TODO: Most of this logic should be pre computed in BANZAI-FLOYDS for performance but this works for now
    orders = orders_from_fits(arc_frame_hdu['ORDER_COEFFS'].data, arc_frame_hdu['ORDER_COEFFS'].header,
                              arc_frame_hdu['SCI'].data.shape)
    wavelengths = WavelengthSolution.from_header(arc_frame_hdu['WAVELENGTHS'].header, orders)
    wavelength_bins = get_wavelength_bins(wavelengths)
    binned_data = bin_data(arc_frame_hdu['SCI'].data, arc_frame_hdu['ERR'].data, wavelengths, orders, wavelength_bins)
    binned_data['background'] = 0.0
    binned_data['weights'] = 1.0
    extracted_data = extract(binned_data)

    fig = make_subplots(rows=2, cols=2, x_title=u'Wavelength (\u212B)', vertical_spacing=0.02,
                        horizontal_spacing=0.05, shared_xaxes=True)

    fig.update_yaxes(title_text='Flux (counts)', row=1, col=1)
    fig.update_yaxes(title_text='Residuals (\u212B)', row=2, col=1)

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
        residuals_wavelengths, residuals = calculate_residuals(wavelength,
                                                               extracted_data['fluxraw'][where_order],
                                                               extracted_data['fluxrawerr'][where_order],
                                                               arc_lines)

        residual_hover_text = [f'{line["wavelength"]:0.3f} {line["line_source"]}' for line in arc_lines
                               if line['wavelength'] in residuals_wavelengths]
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
    fig.update_layout(showlegend=False, autosize=True, margin=dict(l=0, r=0, t=0, b=0))
    line_plot = dcc.Graph(id='arc-line-graph', figure=fig, style={'display': 'inline-block',
                                                                  'width': '100%', 'height': '550px;'})
    return line_plot


def make_2d_sci_plot(frame):
    zmin, zmax = np.percentile(frame['SCI'].data, [1, 99])
    trace = go.Heatmap(z=frame['SCI'].data, colorscale=COLORMAP, zmin=zmin, zmax=zmax, hoverinfo='none')

    layout = dict(title='', margin=dict(t=20, b=50, l=50, r=40), height=350, showlegend=False)
    figure_data = [trace]

    orders = orders_from_fits(frame['ORDER_COEFFS'].data,
                              frame['ORDER_COEFFS'].header,
                              frame['SCI'].data.shape)
    wavelengths = WavelengthSolution.from_header(frame['WAVELENGTHS'].header, orders)
    wavelength_bins = get_wavelength_bins(wavelengths)
    binned_profile = bin_data(frame['PROFILE'].data,
                              np.zeros_like(frame['PROFILE'].data),
                              wavelengths, orders, wavelength_bins)
    for order in [2, 1]:
        extract_center = []
        extract_low = []
        extract_high = []

        background_upper_start = []
        background_upper_end = []

        background_lower_start = []
        background_lower_end = []

        plot_x = []

        for profile_to_fit in binned_profile.groups:
            if profile_to_fit['order'][0] != order:
                continue
            plot_x.append(np.mean(profile_to_fit['x']))
            # Fit a simple gaussian to the profile to get the center and width
            best_fit, _ = curve_fit(gauss, profile_to_fit['y'], profile_to_fit['data'],
                                    [profile_to_fit[np.argmax(profile_to_fit['data'])]['y'],
                                     5.0], method='lm')
            extract_center.append(best_fit[0])
            extract_high.append(best_fit[0] + 2.0 * best_fit[1])
            extract_low.append(best_fit[0] - 2.0 * best_fit[1])

            background_upper_start.append(best_fit[0] + 3.0 * best_fit[1])
            background_lower_start.append(best_fit[0] - 3.0 * best_fit[1])

            background_upper_end.append(np.max(profile_to_fit['y']))
            background_lower_end.append(np.min(profile_to_fit['y']))

        figure_data.append(go.Scatter(x=plot_x, y=extract_center,
                                      mode='lines', line={'color': 'salmon'},
                                      hoverinfo='skip'))
        figure_data.append(go.Scatter(x=plot_x, y=extract_high,
                                      mode='lines', line={'color': 'salmon', 'dash': 'dash'},
                                      hoverinfo='skip'))
        figure_data.append(go.Scatter(x=plot_x, y=extract_low,
                                      mode='lines', line={'color': 'salmon', 'dash': 'dash'},
                                      hoverinfo='skip'))
        figure_data.append(go.Scatter(x=plot_x, y=background_lower_start,
                                      mode='lines', line={'color': '#8F0B0B', 'dash': 'dash'},
                                      hoverinfo='skip'))
        figure_data.append(go.Scatter(x=plot_x, y=background_lower_end,
                                      mode='lines', line={'color': '#8F0B0B', 'dash': 'dash'},
                                      hoverinfo='skip'))
        figure_data.append(go.Scatter(x=plot_x, y=background_upper_start,
                                      mode='lines', line={'color': '#8F0B0B', 'dash': 'dash'},
                                      hoverinfo='skip'))
        figure_data.append(go.Scatter(x=plot_x, y=background_upper_end,
                                      mode='lines', line={'color': '#8F0B0B', 'dash': 'dash'},
                                      hoverinfo='skip'))

    fig = dict(data=figure_data, layout=layout)
    image_plot = dcc.Graph(id='sci-2d-graph', figure=fig, style={'display': 'inline-block',
                                                                 'width': '100%', 'height': '550px;'})
    return image_plot


def unfilled_histogram(x, y, color):
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
    return go.Scatter(x=x_plot, y=y_plot, mode='lines', line={'color': color}, hoverinfo='skip')


def make_profile_plot(sci_2d_frame):
    layout = dict(title='', margin=dict(t=20, b=50, l=50, r=40), height=350, showlegend=False)
    fig = make_subplots(rows=1, cols=4, vertical_spacing=0.02)
    plot_column = {2: 1, 1: 3}

    orders = orders_from_fits(sci_2d_frame['ORDER_COEFFS'].data,
                              sci_2d_frame['ORDER_COEFFS'].header,
                              sci_2d_frame['SCI'].data.shape)
    # Approximate order center to plot in x pixels
    order_center = {2: 1000, 1: 1000}
    x2d, y2d = np.meshgrid(np.arange(sci_2d_frame['SCI'].data.shape[1]), np.arange(sci_2d_frame['SCI'].data.shape[0]))
    for order in [2, 1]:
        where_order = orders.data == order
        y_region = y2d[np.logical_and(x2d == order_center[order], where_order)]
        y_range = slice(np.min(y_region) + 2, np.max(y_region) - 2)

        x_range = slice(order_center[order] - 5, order_center[order] + 5)
        # Take a +-5 region in x around a central part of the order and median to reject cosmic rays
        data = sci_2d_frame['SCI'].data[y_range, x_range]
        data = np.median(data, axis=1)

        # Do the same on the profile extension to show the model
        profile = sci_2d_frame['PROFILE'].data[y_range, x_range]
        profile = np.median(profile, axis=1)

        x_plot = y2d[y_range, order_center[order]] - orders.center(order_center[order])[order - 1]

        fig.add_trace(
            unfilled_histogram(x_plot, data, '#023858'),
            row=1, col=plot_column[order],
        )

        fig.add_trace(
            unfilled_histogram(x_plot, profile, 'salmon'),
            row=1, col=plot_column[order],
        )

        wavelengths = WavelengthSolution.from_header(sci_2d_frame['WAVELENGTHS'].header, orders)
        wavelength_bins = get_wavelength_bins(wavelengths)
        # Calculate the center of the profile per wavelength bin
        binned_data = bin_data(sci_2d_frame['SCI'].data,
                               sci_2d_frame['ERR'].data,
                               wavelengths, orders, wavelength_bins)

        # Compare to the center of mass of the model
        binned_profile = bin_data(sci_2d_frame['PROFILE'].data,
                                  np.zeros_like(sci_2d_frame['PROFILE'].data),
                                  wavelengths, orders, wavelength_bins)

        x_plot = []
        y_profile_plot = []
        y_plot = []
        for data_to_fit, profile_to_fit in zip(binned_data.groups, binned_profile.groups):
            if data_to_fit['order'][0] != order:
                continue
            # Fit a simple gaussian to the profile to get the center and width
            best_fit, _ = curve_fit(gauss, data_to_fit['y'], profile_to_fit['data'],
                                    [profile_to_fit[np.argmax(profile_to_fit['data'])]['y'],
                                     5.0], method='lm')
            x_plot.append(np.mean(profile_to_fit['x']))
            y_profile_plot.append(best_fit[0])
            # Then refit the data with width from the profile in the same way the pipeline does
            # TODO: This should really just be stored in the original data file
            initial_guess = (data_to_fit['y'][np.argmax(data_to_fit['data'])],)
            best_fit_center, = optimize_match_filter(initial_guess, data_to_fit['data'],
                                                     data_to_fit['uncertainty'],
                                                     profile_gauss_fixed_width,
                                                     data_to_fit['y'],
                                                     args=(best_fit[1],))
            y_plot.append(best_fit_center)

        fig.add_trace(
            go.Scatter(x=x_plot, y=y_plot, mode='markers', marker={'color': '#023858'},
                       hoverinfo='skip'),
            row=1, col=plot_column[order] + 1,
            )
        fig.add_trace(
            go.Scatter(x=x_plot, y=y_profile_plot, mode='lines', line={'color': 'salmon'},
                       hoverinfo='skip'),
            row=1, col=plot_column[order] + 1,
            )
    fig.update_layout(**layout)
    profile_plot = dcc.Graph(id='profile-graph', figure=fig,
                             style={'display': 'inline-block', 'width': '100%', 'height': '100%;'})
    return profile_plot


def make_1d_sci_plot(frame_id, archive_header):
    layout = dict(title='', margin=dict(t=0, b=50, l=0, r=0), height=1050, showlegend=False)
    frame_1d = download_frame(url=f'{settings.ARCHIVE_URL}{frame_id}/', headers=archive_header)[1].data
    fig = make_subplots(rows=3, cols=2, x_title=u'Wavelength (\u212B)', vertical_spacing=0.02, horizontal_spacing=0.07,
                        shared_xaxes=True)
    plot_column = {2: 1, 1: 2}
    for order in [2, 1]:
        where_order = frame_1d['order'] == order
        fig.add_trace(
            go.Scatter(x=frame_1d['wavelength'][where_order], y=frame_1d['flux'][where_order],
                       line_color='#023858', mode='lines'),
            row=1, col=plot_column[order],
        )
        fig.add_trace(
            go.Scatter(x=frame_1d['wavelength'][where_order], y=frame_1d['fluxraw'][where_order],
                       line_color='#023858', mode='lines'),
            row=2, col=plot_column[order],
        )
        fig.add_trace(
            go.Scatter(x=frame_1d['wavelength'][where_order], y=frame_1d['background'][where_order],
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
    arc_frame = get_related_frame(frame_id, archive_header, 'L1IDARC')
    arc_image_plot = make_arc_2d_plot(arc_frame)
    arc_line_plot = make_arc_line_plots(arc_frame)

    sci_2d_frame = get_related_frame(frame_id, archive_header, 'L1ID2D')
    sci_2d_plot = make_2d_sci_plot(sci_2d_frame)

    profile_plot = make_profile_plot(sci_2d_frame)
    sci_1d_plot = make_1d_sci_plot(frame_id, archive_header)
    # Return the plot as the child of the output container div
    return [arc_image_plot, arc_line_plot, sci_2d_plot, profile_plot, sci_1d_plot]
