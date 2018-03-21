from cycler import cycler

__DARK_MPL_THEME = [
    ('backend', 'module://ipykernel.pylab.backend_inline'),
    ('interactive', True),
    ('lines.color', '#5050f0'),
    ('text.color', '#d0d0d0'),
    ('axes.facecolor', '#151515'),
    ('axes.edgecolor', '#404040'),
    ('axes.grid', True),
    ('axes.labelsize', 'large'),
    ('axes.labelcolor', 'green'),
    ('axes.prop_cycle', cycler('color', ['#449AcD', 'g', '#f62841', 'y', '#088487', '#E24A33', '#f01010'])),
    ('legend.fontsize', 'small'),
    ('legend.fancybox', False),
    ('legend.edgecolor', '#305030'),
    ('legend.shadow', False),
    ('xtick.color', '#509050'),
    ('ytick.color', '#509050'),
    ('grid.color', '#404040'),
    ('grid.linestyle', '--'),
    ('grid.linewidth', 0.5),
    ('grid.alpha', 0.8),
    ('figure.figsize', [8.0, 5.0]),
    ('figure.dpi', 80.0),
    ('figure.facecolor', (1, 1, 1, 0)),
    ('figure.edgecolor', (1, 1, 1, 0)),
    ('figure.subplot.bottom', 0.125)
]

__LIGHT_MPL_THEME = [
    ('backend', 'module://ipykernel.pylab.backend_inline'),
    ('interactive', True),
    ('lines.color', '#101010'),
    ('text.color', '#303030'),
    ('lines.antialiased', True),
    ('lines.linewidth', 1),
    ('patch.linewidth', 0.5),
    ('patch.facecolor', '#348ABD'),
    ('patch.edgecolor', '#eeeeee'),
    ('patch.antialiased', True),
    ('axes.facecolor', '#fafafa'),
    ('axes.edgecolor', '#d0d0d0'),
    ('axes.linewidth', 1),
    ('axes.titlesize', 'x-large'),
    ('axes.labelsize', 'large'),
    ('axes.labelcolor', '#555555'),
    ('axes.axisbelow', True),
    ('axes.grid', True),
    ('axes.prop_cycle', cycler('color', ['#6792E0', '#27ae60', '#c44e52', '#975CC3', '#ff914d', '#77BEDB',
                                         '#303030', '#4168B7', '#93B851', '#e74c3c', '#bc89e0', '#ff711a',
                                         '#3498db', '#6C7A89'])),
    ('legend.fontsize', 'small'),
    ('legend.fancybox', False),
    ('xtick.color', '#707070'),
    ('ytick.color', '#707070'),
    ('grid.color', '#606060'),
    ('grid.linestyle', '--'),
    ('grid.linewidth', 0.5),
    ('grid.alpha', 0.3),
    ('figure.figsize', [8.0, 5.0]),
    ('figure.dpi', 80.0),
    ('figure.facecolor', '#ffffff'),
    ('figure.edgecolor', '#ffffff'),
    ('figure.subplot.bottom', 0.1)
]


def set_mpl_theme(theme_name='light'):
    import matplotlib
    if 'dark' in theme_name.lower():
        for (k, v) in __DARK_MPL_THEME:
            matplotlib.rcParams[k] = v
    elif 'light' in theme_name.lower():
        for (k, v) in __LIGHT_MPL_THEME:
            matplotlib.rcParams[k] = v
