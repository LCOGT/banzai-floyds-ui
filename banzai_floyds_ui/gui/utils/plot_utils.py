import numpy as np


DARK_BLUE = '#023858'
COLORMAP = [
    [0, '#fff7fb'],
    [0.125, '#ece7f2'],
    [0.25, '#d0d1e6'],
    [0.375, '#a6bddb'],
    [0.5, '#74a9cf'],
    [0.625, '#3690c0'],
    [0.75, '#0570b0'],
    [0.875, '#045a8d'],
    [1, DARK_BLUE]
]
DARK_SALMON = '#8F0B0B'
LAVENDER = '#BB69F5'


def xy_to_svg_path(x, y):
    # We don't want a Z at the end because we're not closing the path
    return 'M ' + ' L '.join(f'{i},{j}' for i, j in zip(x, y))


def unfilled_histogram(x, y, color, name=None, legend=None, axis=''):
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
    if axis == 1:
        axis = ''
    return dict(type='scatter', x=x_plot, y=y_plot, mode='lines', line={'color': color}, hoverinfo='skip',
                name=name, showlegend=show_legend, legend=legend, xaxis=f'x{axis}', yaxis=f'y{axis}')


def json_to_polynomial(poly_info):
    return np.polynomial.legendre.Legendre(coef=poly_info['coef'], domain=poly_info['domain'])


EXTRACTION_REGION_LINE_ORDER = ['center', 'extract_lower', 'extract_upper', 'bkg_left_inner', 'bkg_right_inner',
                                'bkg_left_outer', 'bkg_right_outer']


def extraction_region_traces(order_polynomial, center_polynomial, width_polynomial,
                             wavelengths_polynomial, extract_lower, extract_upper,
                             bkg_left_inner, bkg_right_inner, bkg_left_outer, bkg_right_outer,
                             center_delta=0.0):
    x = np.arange(wavelengths_polynomial.domain[0], wavelengths_polynomial.domain[1] + 1)
    wavelengths = wavelengths_polynomial(x)
    extract_sigma = width_polynomial(wavelengths)
    extract_center = center_polynomial(wavelengths)

    extract_center += order_polynomial(x)
    extract_center += center_delta
    extract_high = extract_center + extract_upper * extract_sigma
    extract_low = extract_center + extract_lower * extract_sigma
    background_upper_start = extract_center + bkg_right_inner * extract_sigma
    background_lower_start = extract_center + bkg_left_inner * extract_sigma
    # Let's start with 10 sigma for now as the edge rather than using the whole slit
    background_lower_end = extract_center + bkg_left_outer * extract_sigma
    background_upper_end = extract_center + bkg_right_outer * extract_sigma
    return x, [extract_center, extract_low, extract_high, background_lower_start, background_upper_start,
               background_lower_end, background_upper_end]


def plot_extracted_data(frame_data):
    figure_data = []
    plot_column = {2: 1, 1: 2}
    for order in [2, 1]:
        where_order = frame_data['order'] == order
        top_row_axis = '' if plot_column[order] == 1 else plot_column[order]
        figure_data.append(
            dict(type='scatter', x=frame_data['wavelength'][where_order], y=frame_data['flux'][where_order],
                 line=dict(color=DARK_BLUE), mode='lines', xaxis=f'x{top_row_axis}', yaxis=f'y{top_row_axis}')
        )
        mid_row_axis = plot_column[order] + 2
        figure_data.append(
            dict(type='scatter', x=frame_data['wavelength'][where_order], y=frame_data['fluxraw'][where_order],
                 line=dict(color=DARK_BLUE), mode='lines', xaxis=f'x{mid_row_axis}', yaxis=f'y{mid_row_axis}')
        )
        bottom_row_axis = plot_column[order] + 4
        figure_data.append(
            dict(type='scatter', x=frame_data['wavelength'][where_order], y=frame_data['background'][where_order],
                 line=dict(color=DARK_BLUE), mode='lines', xaxis=f'x{bottom_row_axis}', yaxis=f'y{bottom_row_axis}')
        )
    return figure_data
