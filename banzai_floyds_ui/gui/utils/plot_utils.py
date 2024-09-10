def xy_to_svg_path(x, y):
    # We don't want a Z at the end because we're not closing the path
    return 'M ' + ' L '.join(f'{i},{j}' for i, j in zip(x, y))