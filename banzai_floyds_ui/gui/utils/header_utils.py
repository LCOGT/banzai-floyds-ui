import numpy as np


def header_to_polynomial(header, key_pattern, order, max_coef=10):
    domain = header[f'O{order}{key_pattern}DM0'], header[f'O{order}{key_pattern}DM1']
    coef = []
    for i in range(max_coef):
        key = f'O{order}{key_pattern}{i:02}'
        if key in header:
            coef.append(header[key])
    return np.polynomial.Legendre(coef, domain=domain)
