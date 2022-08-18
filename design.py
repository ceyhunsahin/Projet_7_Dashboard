# the style arguments for the sidebar. We use position:fixed and a fixed width
SIDEBAR_STYLE = {
    "position": "fixed",
    "top": 12,
    "left": 0,
    "bottom": 0,
    "width": "21rem",
    "height": "100%",
    "z-index": 1,
    "overflow-x": "hidden",
    "transition": "all 1s",
    "padding": "0.5rem 1rem",
    "background-color": "#f8f9fa",
}

SIDEBAR_HIDEN = {
    "position": "fixed",
    "top": 12,
    "left": "-21rem",
    "bottom": 0,
    "width": "21rem",
    "height": "100%",
    "z-index": 1,
    "overflow-x": "hidden",
    "transition": "all 1s",
    "padding": "0rem 0rem",
    "background-color": "#f8f9fa",
}

# the styles for the main content position it to the right of the sidebar and
# add some padding.
CONTENT_STYLE = {
    "transition": "margin-left .5s",
    "margin-left": "30rem",
    "margin-right": "2rem",
    "padding": "2rem 1rem",
    "width": "70%",
    "transition": "all 1s"
    # "background-color": "#f8f9fa",
}

CONTENT_STYLE_client = {
    "margin-left": "30rem",
    "margin-right": "2rem",
    "padding": "2rem 1rem",
    "width": "70%",
    'visibility': 'hidden',
    "transition": "all 1s"
    # "background-color": "#f8f9fa",
}

CONTENT_STYLE1 = {
    "transition": "margin-left .5s",
    "margin-left": "1rem",
    "margin-right": "2rem",
    "padding": "2rem 1rem",
    "width": "100%",
    "transition": "all 1s"
    # "background-color": "#f8f9fa",
}

CONTENT_STYLE1_client = {
    "transition": "margin-left .5s",
    "margin-left": "1rem",
    "margin-right": "2rem",
    "padding": "2rem 1rem",
    "width": "100%",
    "transition": "all 1s"
    # "background-color": "#f8f9fa",
}