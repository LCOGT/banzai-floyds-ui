from banzai_floyds_ui.gui.utils.header_utils import header_to_polynomial

import numpy as np
import plotly.graph_objs as go
from banzai_floyds.orders import orders_from_fits
from banzai_floyds.utils.wavelength_utils import WavelengthSolution
from dash import dcc
from scipy.interpolate import LinearNDInterpolator

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


def make_2d_sci_plot(frame, filename):
    zmin, zmax = np.percentile(frame['SCI'].data, [1, 99])
    trace = go.Heatmap(z=frame['SCI'].data, colorscale=COLORMAP, zmin=zmin, zmax=zmax, hoverinfo='none',
                       colorbar=dict(title='Data (counts)'))

    layout = dict(title=f'2-D Science Frame: {filename}', margin=dict(t=40, b=50, l=50, r=40),
                  height=370)
    layout['legend'] = dict(x=0, y=0.95)
    layout['xaxis'] = dict(title='x (pixel)')
    layout['yaxis'] = dict(title='y (pixel)')
    layout['shapes'] = []
    figure_data = [trace]
    for order in [2, 1]:
        orders = orders_from_fits(frame['ORDER_COEFFS'].data, frame['ORDER_COEFFS'].header,
                                  frame['SCI'].data.shape)
        this_order = orders.data == order
        x2d, y2d = np.meshgrid(np.arange(frame['SCI'].data.shape[1]), np.arange(frame['SCI'].data.shape[0]))
        # Invert the wavelength solution with the y of the profile to get the x of the plot
        wavelengths2d = WavelengthSolution.from_header(frame['WAVELENGTHS'].header, orders)
        wavelengths = np.linspace(wavelengths2d.data[this_order].min(),
                                  wavelengths2d.data[this_order].max(),
                                  num=int(np.max(x2d[this_order]) - np.min(x2d[this_order])))
        center_polynomial = header_to_polynomial(frame['PROFILEFITS'].header, 'CTR', order)
        width_polynomal = header_to_polynomial(frame['PROFILEFITS'].header, 'WID', order)

        extract_sigma = width_polynomal(wavelengths)
        x_interpolator = LinearNDInterpolator((y2d[this_order] - orders.center(x2d[this_order])[order - 1],
                                               wavelengths2d.data[this_order]), x2d[this_order])
        extract_center = center_polynomial(wavelengths)
        x = x_interpolator(extract_center, wavelengths)
        extract_center = extract_center[np.isfinite(x)]
        extract_sigma = extract_sigma[np.isfinite(x)]
        x = x[np.isfinite(x)]
        extract_center += orders.center(x)[order - 1]
        # Right now we just N sigma defined from the header to set the extraction lines
        # We really should just use the extraction_window values in the binned data
        # TODO: We need to store the binned data

        n_extract_sigma = frame['SCI'].header['EXTRTWIN']
        extract_high = extract_center + n_extract_sigma * extract_sigma
        extract_low = extract_center - n_extract_sigma * extract_sigma

        # We currently use N sigma from the profile center to set the background region
        # TODO: use a background_window in the binned data analogous to the extraction_window
        bkg_lower_n_sigma = frame['SCI'].header['BKWINDW0']
        bkg_upper_n_sigma = frame['SCI'].header['BKWINDW1']
        background_upper_start = extract_center + bkg_lower_n_sigma * extract_sigma
        background_lower_start = extract_center - bkg_lower_n_sigma * extract_sigma
        # Let's start with 10 sigma for now as the edge rather than using the whole slit
        background_lower_end = extract_center - bkg_upper_n_sigma * extract_sigma
        background_upper_end = extract_center + bkg_upper_n_sigma * extract_sigma

        if order == 2:
            center_name = 'Extraction Center'
            center_legend = True

            extraction_name = f'Extraction \u00b1{n_extract_sigma:0.1f}\u03C3'
            extreaction_legend = True

            background_name = 'Background Region'
            background_legend = True
        else:
            center_name = None
            center_legend = False

            extraction_name = None
            extreaction_legend = False

            background_name = None
            background_legend = False

        figure_data.append(go.Scatter(x=x, y=extract_center, mode='lines',
                                      line={'color': LAVENDER, 'width': 2}, hoverinfo='skip',
                                      name=center_name, showlegend=center_legend))
        figure_data.append(go.Scatter(x=x, y=extract_high,
                                      line={'color': LAVENDER, 'dash': 'dash', 'width': 2},
                                      mode='lines', hoverinfo='skip',
                                      name=extraction_name, showlegend=extreaction_legend))
        figure_data.append(go.Scatter(x=x, y=extract_low,
                                      line={'color': LAVENDER, 'dash': 'dash'},
                                      mode='lines', hoverinfo='skip',
                                      showlegend=False))
        figure_data.append(go.Scatter(x=x, y=background_lower_start,
                                      line={'color': DARK_SALMON, 'dash': 'dash'},
                                      mode='lines', hoverinfo='skip',
                                      name=background_name, showlegend=background_legend))
        figure_data.append(go.Scatter(x=x, y=background_lower_end,
                                      line={'color': DARK_SALMON, 'dash': 'dash'},
                                      mode='lines', hoverinfo='skip',
                                      showlegend=False))
        figure_data.append(go.Scatter(x=x, y=background_upper_start,
                                      line={'color': DARK_SALMON, 'dash': 'dash'},
                                      mode='lines', hoverinfo='skip',
                                      showlegend=False))
        figure_data.append(go.Scatter(x=x, y=background_upper_end,
                                      line={'color': DARK_SALMON, 'dash': 'dash'},
                                      mode='lines', hoverinfo='skip',
                                      showlegend=False))

    fig = dict(data=figure_data, layout=layout)
    image_plot = dcc.Graph(id='sci-2d-graph', figure=fig, style={'display': 'inline-block',
                                                                 'width': '100%', 'height': '550px;'})
    return image_plot
