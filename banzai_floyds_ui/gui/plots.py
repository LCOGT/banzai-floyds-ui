import importlib.resources
from astropy.table import Table
from django.conf import settings
from banzai_floyds_ui.gui.utils.file_utils import download_frame
from banzai_floyds_ui.gui.utils.header_utils import header_to_polynomial

import numpy as np
from numpy.polynomial.legendre import Legendre
from banzai_floyds.orders import orders_from_fits
from banzai_floyds.arc_lines import used_lines as arc_lines
from banzai_floyds.utils.wavelength_utils import WavelengthSolution
from scipy.interpolate import LinearNDInterpolator
from banzai_floyds_ui.gui.utils.plot_utils import unfilled_histogram, EXTRACTION_REGION_LINE_ORDER
from banzai_floyds_ui.gui.utils.plot_utils import extraction_region_traces, plot_extracted_data
import importlib
import json


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
DARK_SALMON = '#8F0B0B'
LAVENDER = '#BB69F5'

template_path = importlib.resources.files('banzai_floyds_ui.gui').joinpath('data/plotly_template.json')
PLOTLY_TEMPLATE = json.loads(template_path.read_text())


def make_2d_sci_plot(frame, filename):
    zmin, zmax = np.percentile(frame['SCI'].data, [1, 99])

    trace = dict(type='heatmap', z=frame['SCI'].data, colorscale=COLORMAP, zmin=zmin, zmax=zmax,
                 hoverinfo='none', colorbar=dict(title='Data (counts)'))

    layout = dict(title=f'2-D Science Frame: {filename}', margin=dict(t=40, b=50, l=50, r=40),
                  height=370, template=PLOTLY_TEMPLATE)
    layout['legend'] = dict(x=0, y=0.95)
    layout['xaxis'] = dict(title='x (pixel)')
    layout['yaxis'] = dict(title='y (pixel)')
    layout['shapes'] = []
    figure_data = [trace]
    trace_info_to_store = {
        'order_center': {
            "1": {},
            "2": {}
        },
        'profile_center': {
            "1": {},
            "2": {}
        },
        'profile_sigma': {
            "1": {},
            "2": {},
        },
        'wavelength': {
            "1": {},
            "2": {}
        }
    }

    for order in [2, 1]:
        orders = orders_from_fits(frame['ORDER_COEFFS'].data, frame['ORDER_COEFFS'].header,
                                  frame['SCI'].data.shape)
        order_polynomial = np.polynomial.Legendre(orders.coeffs[order - 1], domain=orders.domains[order - 1])

        wavelenth_solution = WavelengthSolution.from_header(frame['WAVELENGTHS'].header, orders)
        wavelengths_polynomial = Legendre(coef=wavelenth_solution.coefficients[order - 1],
                                          domain=wavelenth_solution.domains[order - 1])
        center_polynomial = header_to_polynomial(frame['PROFILEFITS'].header, 'CTR', order)
        width_polynomal = header_to_polynomial(frame['PROFILEFITS'].header, 'WID', order)
        for polynomial, key in zip([order_polynomial, center_polynomial, width_polynomal, wavelengths_polynomial],
                                   ['order_center', 'profile_center', 'profile_sigma', 'wavelength']):
            for attribute in ['coef', 'domain']:
                trace_info_to_store[key][str(order)][attribute] = getattr(polynomial, attribute)

        # Right now we just N sigma defined from the header to set the extraction lines
        # We really should just use the extraction_window values in the binned data
        # We currently use N sigma from the profile center to set the background region
        # TODO: use a background_window in the binned data analogous to the extraction_window

        extract_lower_n_sigma = frame['SCI'].header[f'XTRTW{order}0']
        upper_lower_n_sigma = frame['SCI'].header[f'XTRTW{order}1']

        bkg_left_lower_n_sigma = frame['SCI'].header[f'BKWO{order}00']
        bkg_left_upper_n_sigma = frame['SCI'].header[f'BKWO{order}01']

        bkg_right_lower_n_sigma = frame['SCI'].header[f'BKWO{order}10']
        bkg_right_upper_n_sigma = frame['SCI'].header[f'BKWO{order}11']

        x, traces = extraction_region_traces(order_polynomial, center_polynomial, width_polynomal,
                                             wavelengths_polynomial, extract_lower_n_sigma, upper_lower_n_sigma,
                                             bkg_left_lower_n_sigma, bkg_right_lower_n_sigma, bkg_left_upper_n_sigma,
                                             bkg_right_upper_n_sigma)

        if order == 2:
            center_name = 'Extraction Center'
            center_legend = True

            extraction_name = 'Extraction Region'
            extraction_legend = True

            background_name = 'Background Region'
            background_legend = True
        else:
            center_name = None
            center_legend = False

            extraction_name = None
            extraction_legend = False

            background_name = None
            background_legend = False

        colors = [LAVENDER, LAVENDER, LAVENDER, DARK_SALMON, DARK_SALMON, DARK_SALMON, DARK_SALMON]
        show_legends = [center_legend, extraction_legend, False, background_legend, False, False, False]
        names = [center_name, extraction_name, None, background_name, None, None, None]
        dasheds = ['solid', 'dash', 'dash', 'dash', 'dash', 'dash', 'dash']
        for y, color, show_legend, name, dashed in zip(traces, colors, show_legends, names, dasheds):
            figure_data.append(dict(type='scatter', x=x, y=y, mode='lines',
                                    line={'color': color, 'width': 2, 'dash': dashed}, hoverinfo='skip',
                                    name=name, showlegend=show_legend))

    figure = dict(data=figure_data, layout=layout)

    return figure, trace_info_to_store


def make_arc_2d_plot(arc_frame_hdu, arc_filename):
    zmin, zmax = np.percentile(arc_frame_hdu['SCI'].data, [1, 99])
    trace = dict(type='heatmap', z=arc_frame_hdu['SCI'].data, colorscale=COLORMAP, zmin=zmin, zmax=zmax,
                 hoverinfo='none', colorbar=dict(title='Data (counts)'))

    layout = dict(margin=dict(t=50, b=50, l=50, r=40), height=370, template=PLOTLY_TEMPLATE)

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
                figure_data.append(dict(
                    type='scatter', x=plot_x, y=y_line,
                    marker={'color': 'salmon'},
                    mode='lines',
                    hovertext=[f'{line["wavelength"]:0.3f} {line["line_source"]}'
                               for _ in range(len(plot_x))],
                    hovertemplate='%{hovertext}<extra></extra>',
                    showlegend=show_legend,
                    name=name
                ))
    layout['legend'] = dict(x=0, y=0.95)
    layout['title'] = f'Arc Frame Used in Reduction: {arc_filename}'
    layout['xaxis'] = dict(title='x (pixel)')
    layout['yaxis'] = dict(title='y (pixel)')
    return dict(data=figure_data, layout=layout)


def make_arc_line_plots(arc_frame_hdu):
    """Make a plot for each order showing the arc lines and their residulas"""

    # We generate the subplots layout manually to make passing things easier between the front and backend
    # Note the origin is in the top left corner for plotly for some reason
    # This is equivalent rows=2, cols=2, vertical_spacing=0.02, horizontal_spacing=0.05, shared_xaxes=True
    layout = {
        'showlegend': False, 'autosize': True, 'margin': dict(l=0, r=0, t=0, b=0), 'template': PLOTLY_TEMPLATE,
        'xaxis': {'anchor': 'y', 'domain': [0.0, 0.475], 'matches': 'x3', 'showticklabels': False},
        'xaxis2': {'anchor': 'y2', 'domain': [0.525, 1.0], 'matches': 'x4', 'showticklabels': False},
        'xaxis3': {'anchor': 'y3', 'domain': [0.0, 0.475]},
        'xaxis4': {'anchor': 'y4', 'domain': [0.525, 1.0]},
        'yaxis': {'anchor': 'x', 'domain': [0.51, 1.0]},
        'yaxis2': {'anchor': 'x2', 'domain': [0.51, 1.0]},
        'yaxis3': {'anchor': 'x3', 'domain': [0.0, 0.49]},
        'yaxis4': {'anchor': 'x4', 'domain': [0.0, 0.49]}
    }
    layout['yaxis']['title'] = {'text': 'Flux (counts)'}
    layout['yaxis3']['title'] = {'text': 'Residuals (\u212B)'}
    layout['xaxis3']['title'] = {'text': 'Wavelength (\u212B)'}
    layout['xaxis4']['title'] = {'text': 'Wavelength (\u212B)'}
    layout['xaxis3']['tickformat'] = '.0f'
    layout['xaxis4']['tickformat'] = '.0f'

    annotations = [dict(xref='x domain', yref='y domain', x=0.01, y=0.97, text='Blue Order (order=2)', showarrow=False),
                   dict(xref='x2 domain', yref='y2 domain', x=0.01, y=0.97, text='Red Order (order=1)', showarrow=False)]
    layout['annotations'] = annotations
    figure_data = []
    extracted_data = arc_frame_hdu['EXTRACTED'].data
    lines_used = arc_frame_hdu['LINESUSED'].data
    plot_column = {2: 1, 1: 2}
    for order in [2, 1]:
        where_order = extracted_data['order'] == order
        if plot_column[order] == 1:
            top_row_axis = ''
        else:
            top_row_axis = '2'
        figure_data.append(
            dict(type='scatter', x=extracted_data['wavelength'][where_order].tolist(),
                 y=extracted_data['fluxraw'][where_order].tolist(),
                 line={'color': COLORMAP[-1][1]}, mode='lines', xaxis=f'x{top_row_axis}', yaxis=f'y{top_row_axis}')
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
            figure_data.append(
                dict(
                    type='scatter',
                    x=annotation_x.tolist(),
                    y=annotation_y.tolist(),
                    hovertext=annotation_hovertext,
                    mode='lines',
                    marker={'color': 'salmon'},
                    hovertemplate='%{hovertext}<extra></extra>',
                    xaxis=f'x{top_row_axis}',
                    yaxis=f'y{top_row_axis}'
                )
            )
        lines_used = arc_frame_hdu['LINESUSED'].data[arc_frame_hdu['LINESUSED'].data['order'] == order]
        reference_wavelengths = lines_used['reference_wavelength']

        residual_hover_text = [f'{line["wavelength"]:0.3f} {line["line_source"]}' for line in arc_lines
                               if line['wavelength'] in reference_wavelengths]
        residuals_wavelengths = lines_used['measured_wavelength']
        residuals = residuals_wavelengths - reference_wavelengths
        figure_data.append(
            dict(
                type='scatter', x=residuals_wavelengths.tolist(), y=residuals.tolist(),
                mode='markers', marker=dict(color='#023858'),
                hovertext=residual_hover_text,
                hovertemplate='%{y}\u212B: %{hovertext}<extra></extra>',
                xaxis=f'x{plot_column[order] + 2}', yaxis=f'y{plot_column[order] + 2}'
            )
        )

        residual_range = np.max(residuals) - np.min(residuals)
        layout[f'yaxis{plot_column[order] + 2}']['range'] = [np.min(residuals) - 0.1 * residual_range,
                                                             np.max(residual_range) + 0.1 * residual_range]

    return {'data': figure_data, 'layout': layout}


def make_profile_plot(sci_2d_frame):
    # This is equivalent to
    # make_subplots(rows=2, cols=2, vertical_spacing=0.13,
    #    subplot_titles=("Profile Cross Section: Blue Order (order=2)", "Profile Center: Blue Order (order=2)",
    #        "Profile Cross Section: Red Order (order=1)", "Profile Center: Red Order (order=1)"))
    annotations = [
        {
            'font': {'size': 16}, 'showarrow': False,
            'text': 'Profile Cross Section: Blue Order (order=2)',
            'x': 0.225, 'xanchor': 'center', 'xref': 'paper',
            'y': 1.0, 'yanchor': 'bottom', 'yref': 'paper'
        },
        {
            'font': {'size': 16}, 'showarrow': False,
            'text': 'Profile Center: Blue Order (order=2)',
            'x': 0.775, 'xanchor': 'center', 'xref': 'paper',
            'y': 1.0, 'yanchor': 'bottom', 'yref': 'paper'
        },
        {
            'font': {'size': 16}, 'showarrow': False,
            'text': 'Profile Cross Section: Red Order (order=1)',
            'x': 0.225, 'xanchor': 'center', 'xref': 'paper',
            'y': 0.435, 'yanchor': 'bottom', 'yref': 'paper'
        },
        {
            'font': {'size': 16}, 'showarrow': False,
            'text': 'Profile Center: Red Order (order=1)',
            'x': 0.775, 'xanchor': 'center', 'xref': 'paper',
            'y': 0.435, 'yanchor': 'bottom', 'yref': 'paper'
        }
    ]
    layout = {
        'template': PLOTLY_TEMPLATE,
        'title': {'text': ''},
        'margin': dict(t=20, b=20, l=50, r=40), 'height': 720, 'showlegend': True,
        'annotations': annotations,
        'legend': dict(y=1.0, x=0.005),
        'legend2': dict(y=1.0, x=0.56),
        'xaxis': {'anchor': 'y', 'title': {'text': 'y offset from center (pixel)'}, 'domain': [0.0, 0.45]},
        'xaxis2': {'anchor': 'y2', 'title': {'text': 'x (pixel)'}, 'domain': [0.55, 1.0]},
        'xaxis3': {'anchor': 'y3', 'title': {'text': 'y offset from center (pixel)'}, 'domain': [0.0, 0.45]},
        'xaxis4': {'anchor': 'y4', 'title': {'text': 'x (pixel)'}, 'domain': [0.55, 1.0]},
        'yaxis': {'anchor': 'x', 'title': {'text': 'Normalized Flux'},
                  'range': [-0.1, 1.5], 'domain': [0.565, 1.0]},
        'yaxis2': {'anchor': 'x2', 'title': {'text': 'y (pixel)'}, 'domain': [0.565, 1.0]},
        'yaxis3': {'anchor': 'x3', 'title': {'text': 'Normalized Flux'},
                   'range': [-0.1, 1.5],'domain': [0.0, 0.435]},
        'yaxis4': {'anchor': 'x4', 'title': {'text': 'y (pixel)'}, 'domain': [0.0, 0.435]}
    }

    figure_data = []
    plot_row = {2: 1, 1: 2}
    # Define the coordinate refernce plot manually per order
    reference_axes = {2: 1, 1: 3}
    # Approximate wavelength center to plot the profile
    order_center = {1: 7000, 2: 4500}
    initial_extraction_info = {'positions': {'1': {}, '2': {}}, 'refsigma': {}, 'refcenter': {}}
    shapes = []

    for order in [2, 1]:
        binned_data = Table(sci_2d_frame['BINNED2D'].data).group_by(('order', 'wavelength_bin'))
        # We have to remove the last index here because astropy prepolulates it with the final row in the table so it
        # knows where to start if you add a new group
        wavelength_bins = np.array([binned_data[index]['wavelength'] for index in binned_data.groups.indices[:-1]])
        closest_wavelength_bin = np.argmin(np.abs(wavelength_bins - order_center[order]))
        data = binned_data[binned_data.groups.indices[closest_wavelength_bin]:
                           binned_data.groups.indices[closest_wavelength_bin + 1]]

        if order == 2:
            model_name = 'Model'
            data_name = 'Data'
        else:
            model_name = None
            data_name = None
        figure_data.append(
            unfilled_histogram(data['y_order'], data['data'], '#023858', name=data_name, legend='legend2',
                               axis=2 * plot_row[order] - 1),
        )

        figure_data.append(
            unfilled_histogram(data['y_order'], data['weights'], 'salmon', name=model_name, legend='legend2',
                               axis=2 * plot_row[order] - 1),
        )

        traced_points = sci_2d_frame['PROFILEFITS'].data[sci_2d_frame['PROFILEFITS'].data['order'] == order]
        figure_data.append(
            dict(
                type='scatter', x=traced_points['wavelength'],
                y=traced_points['center'],
                mode='markers', marker={'color': '#023858'},
                hoverinfo='skip', showlegend=False,
                xaxis=f'x{plot_row[order] * 2}', yaxis=f'y{plot_row[order] * 2}'
            )
        )
        center_polynomial = header_to_polynomial(sci_2d_frame['PROFILEFITS'].header, 'CTR', order)
        x_plot = np.arange(center_polynomial.domain[0], center_polynomial.domain[1] + 1, 1.0)
        figure_data.append(
            dict(
                type='scatter', x=x_plot, y=center_polynomial(x_plot), mode='lines', line={'color': 'salmon'},
                hoverinfo='skip', showlegend=False,
                xaxis=f'x{plot_row[order] * 2}', yaxis=f'y{plot_row[order] * 2}'
            )
        )
        # Add in the extraction center and region and background region lines
        # We do this based on header keywords, but we really should do it on the binned data
        extract_center = center_polynomial(order_center[order])
        initial_extraction_info['refcenter'][str(order)] = extract_center

        width_polynomial = header_to_polynomial(sci_2d_frame['PROFILEFITS'].header, 'WID', order)
        extract_sigma = width_polynomial(order_center[order])
        initial_extraction_info['refsigma'][str(order)] = extract_sigma

        extract_lower_n_sigma = sci_2d_frame['SCI'].header[f'XTRTW{order}0']
        extract_upper_n_sigma = sci_2d_frame['SCI'].header[f'XTRTW{order}1']

        bkg_left_lower_n_sigma = sci_2d_frame['SCI'].header[f'BKWO{order}00']
        bkg_left_upper_n_sigma = sci_2d_frame['SCI'].header[f'BKWO{order}01']

        bkg_right_lower_n_sigma = sci_2d_frame['SCI'].header[f'BKWO{order}10']
        bkg_right_upper_n_sigma = sci_2d_frame['SCI'].header[f'BKWO{order}11']

        colors = [LAVENDER, LAVENDER, LAVENDER, DARK_SALMON, DARK_SALMON, DARK_SALMON, DARK_SALMON]
        dashed = ['solid', 'dash', 'dash', 'dash', 'dash', 'dash', 'dash']
        names = ['Extraction Center', 'Extraction Region', 'Extraction Region', 'Background Region',
                 'Background Region', 'Background Region', 'Background Region']

        bkg_right_lower_n_sigma = sci_2d_frame['SCI'].header[f'BKWO{order}10']
        xs = [extract_center + n_sigma * extract_sigma for n_sigma in
              [0, extract_lower_n_sigma, extract_upper_n_sigma, bkg_left_lower_n_sigma, bkg_left_upper_n_sigma,
               bkg_right_lower_n_sigma, bkg_right_upper_n_sigma]]
        for position, key in zip(xs, EXTRACTION_REGION_LINE_ORDER):
            initial_extraction_info['positions'][str(order)][key] = position

        for x, name, color, dash in zip(xs, names, colors, dashed):
            axis_index = "" if reference_axes[order] == 1 else reference_axes[order]
            shapes.append({'type': 'line',
                           'x0': x, 'x1': x,
                           'y0': -2.0, 'y1': 2.0,
                           'name': name,
                           'line': {'color': color, 'width': 2, 'dash': dash},
                           'xref': f'x{axis_index}', 'yref': f'y{axis_index}'})

    # Add dummy traces to make the legend... ugh...
    for name, color, dash in zip(['Extraction Center', 'Extraction Region', 'Background Region'],
                                 [LAVENDER, LAVENDER, DARK_SALMON], ['solid', 'dash', 'dash']):
        figure_data.append(
            dict(
                type='scatter', x=[None], y=[None], mode='lines',
                line={'color': color, 'width': 2, 'dash': dash}, name=name, legend='legend',
                showlegend=True, xaxis='x', yaxis='y'
            )
        )
    layout['shapes'] = shapes
    return {'data': figure_data, 'layout': layout}, initial_extraction_info


def make_1d_sci_plot(frame_id, archive_header):

    frame_1d = download_frame(url=f'{settings.ARCHIVE_URL}{frame_id}/', headers=archive_header)
    frame_data = frame_1d[1].data
    # We again make the layout dict manually. Below is equivilent to
    # make_subplots(rows=3, cols=2, vertical_spacing=0.02, horizontal_spacing=0.07, shared_xaxes=True, 
    # subplot_titles=['Blue Order (order=2)', 'Red Order (order=1)', None, None, None, None])
    title_dict = {
        'text': f"1-D Extractions: {frame_1d[0].header['ORIGNAME'].replace('-e00', '-e91-1d')}",
        'y': 0.985,
        'x': 0.5,
        'xanchor': 'center',
        'yanchor': 'top'
        }
    subplot_titles = [{'font': {'size': 16}, 'showarrow': False, 'text': 'Blue Order (order=2)', 'x': 0.2325,
                       'xanchor': 'center', 'xref': 'paper', 'y': 1.0, 'yanchor': 'bottom', 'yref': 'paper'},
                      {'font': {'size': 16}, 'showarrow': False, 'text': 'Red Order (order=1)', 'x': 0.7675,
                       'xanchor': 'center', 'xref': 'paper', 'y': 1.0, 'yanchor': 'bottom', 'yref': 'paper'}]
    layout = {
        'template': PLOTLY_TEMPLATE,
        'title': title_dict,
        'margin': dict(t=60, b=50, l=0, r=0),
        'height': 1080,
        'showlegend': False,
        'annotations': subplot_titles,
        'xaxis': {'anchor': 'y', 'domain': [0.0, 0.465], 'matches': 'x5', 'showticklabels': False},
        'xaxis2': {'anchor': 'y2', 'domain': [0.535, 1.0], 'matches': 'x6', 'showticklabels': False},
        'xaxis3': {'anchor': 'y3', 'domain': [0.0, 0.465], 'matches': 'x5', 'showticklabels': False},
        'xaxis4': {'anchor': 'y4', 'domain': [0.535, 1.0], 'matches': 'x6', 'showticklabels': False},
        'xaxis5': {'anchor': 'y5', 'domain': [0.0, 0.465], 'tickformat': '.0f',
                   'title': {'text': 'Wavelength (\u212B)'}},
        'xaxis6': {'anchor': 'y6', 'domain': [0.535, 1.0], 'tickformat': '.0f',
                   'title': {'text': 'Wavelength (\u212B)'}},
        'yaxis': {
            'anchor': 'x',
            'domain': [0.68, 1.0],
            'title': {'text': 'Flux (erg s\u207B\u00B9 cm\u207B\u00B2 \u212B\u207B\u00B9)'},
            'exponentformat': 'power'
        },
        'yaxis2': {'anchor': 'x2', 'domain': [0.68, 1.0], 'exponentformat': 'power'},
        'yaxis3': {'anchor': 'x3', 'domain': [0.34, 0.66],
                   'title': {'text': 'Flux (counts)'}, 'exponentformat': 'power'},
        'yaxis4': {'anchor': 'x4', 'domain': [0.34, 0.66], 'exponentformat': 'power'},
        'yaxis5': {
            'anchor': 'x5',
            'domain': [0.0, 0.32],
            'title': {'text': 'Background (counts)'},
            'exponentformat': 'power'
        },
        'yaxis6': {'anchor': 'x6', 'domain': [0.0, 0.32], 'exponentformat': 'power'}
    }

    figure_data = plot_extracted_data(frame_data)

    return {'data': figure_data, 'layout': layout}
