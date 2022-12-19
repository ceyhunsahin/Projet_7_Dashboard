# -*- coding: utf-8 -*-
import base64
import time
import dash
import dash_bootstrap_components as dbc
import dash_daq as daq
import pandas as pd
import numpy as np
import plotly.graph_objs as go
import plotly.express as px
import requests
import shap
from dash import dash_table  # #
from dash import dcc, no_update
from dash import html
from dash.dependencies import Input, Output, State
from dash.exceptions import PreventUpdate
from flask_caching import Cache
import xgboost as xgb
from design import SIDEBAR_STYLE, SIDEBAR_HIDEN,CONTENT_STYLE, CONTENT_STYLE_client,CONTENT_STYLE1,CONTENT_STYLE1_client


# import datasets(normal, normalise) and model
# we use github repo for get the datasets
path = 'https://raw.githubusercontent.com/ceyhunsahin/Projet_7_Dashboard/master/Projet_File/test_sample_data_home_risk.csv'
path2 = 'https://raw.githubusercontent.com/ceyhunsahin/Projet_7_Dashboard/master/Projet_File/test_sample_data_home_risk_normalise.csv'
path3 = 'pipeline_housing.json'

# Sample 800 for df_test
df_test = pd.read_csv(path, encoding='unicode_escape')
df_test = df_test.sample(800, random_state=42)

df_test = df_test.loc[:, ~df_test.columns.str.match ('Unnamed')]
df_test = df_test.sort_values ('SK_ID_CURR')
# Sample 1200 for df_test_normalize
df_test_normalize = pd.read_csv (path2, index_col=0)
df_test_normalize = df_test_normalize.sample(800, random_state=42)

# import model
model = xgb.XGBClassifier ()
model.load_model(path3)

BS = "https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/css/bootstrap.min.css" # bootstrap for design
# Initialize the app
app = dash.Dash (__name__, external_stylesheets=[BS], suppress_callback_exceptions=True, update_title='Loading...',
                 meta_tags=[{ 'name': 'viewport',
                              'content': 'width=device-width, initial-scale=2.0, maximum-scale=1.2, minimum-scale=0.5' },
                            ]
                 )
server = app.server
app.config.suppress_callback_exceptions = True

# add favicon
app._favicon = ("assets/favicon.ico")

# to add SHAP images
image_filename_1 = 'summary_plot3.png'  # replace with your own image
encoded_image_1 = base64.b64encode (open (image_filename_1, 'rb').read ())
image_filename_2 = 'summary_plot4.png'  # replace with your own image
encoded_image_2 = base64.b64encode (open (image_filename_2, 'rb').read ())

# Now, for decrease the dimension of the dashboard, we use Flask-Cache capability
cache = Cache (app.server, config={
    # Note that filesystem cache doesn't work on systems with ephemeral
    # filesystems like Heroku.
    'CACHE_TYPE': 'filesystem',
    'CACHE_DIR': 'cache-directory',
    # should be equal to maximum number of users on the app at a single time
    # higher numbers will store more data in the filesystem / redis cache
    'CACHE_THRESHOLD': 200
})
TIMEOUT = 300

# get request with FLASK API, this function is getting all of the dataset for comparing the clients with the others
@cache.memoize (timeout=TIMEOUT)
def query_data():
    # This approach works well if there is one dataset that is used to update several callbacks.
    url_api_model_result = 'https://oc-p7-home-risk-flaskapi.herokuapp.com/scores'
    get_request = requests.get (url=url_api_model_result, params={ 'index': 100111 })
    total_score = ''
    get_request.raise_for_status ()
    if get_request.status_code != 204:
        total_score = get_request.json ()['Total_score']
    total_score = pd.read_json (total_score, orient='index', convert_axes=False, convert_dates=False).T

    return total_score


# graph capabilities
config = { 'displayModeBar': True,
           'scrollZoom': True,
           'displaylogo': False,
           'modeBarButtonsToAdd': [
               'drawopenpath',
               'drawcircle',
               'eraseshape',
               'select2d',
           ] }

# sidebar design

sidebar = html.Div (
    [
        dbc.Button ("X", outline=True, color="secondary", className="mr-1", id="btn_sidebar",
                    style={ "margin-left": "260px" }),

        html.Img (src="https://user.oc-static.com/upload/2019/02/25/15510866018677_logo%20projet%20fintech.png",
                  alt="Logo entreprise ", height='300rem', width='300rem'),

        html.Hr (),
        html.P (
            "Sélectionner une demande de prêt:", className="lead"
        ),
        # html.P("Prêt ID : ", className="lead"
        #       ),
        dcc.Dropdown (id='pret_id',
                      options=[{ 'label': i, 'value': i } for i in
                               df_test.SK_ID_CURR],
                      multi=False,
                      style={ 'cursor': 'pointer', 'width': '300px' },

                      clearable=True,
                      placeholder='Prét ID...',
                      ),

        dcc.Checklist (id='stades_client',
                       options=[
                           { 'label': 'Données Client', 'value': 'don_client' },
                           { 'label': 'Résultat de la demande de prêt', 'value': 'result_dem' },
                           { 'label': 'Analyse des features client', 'value': 'analyse_client' },
                       ],
                       value=['don_client'], labelStyle={ 'display': 'inline-Block', 'margin': '1rem', },
                       inputStyle={ "marginRight": "20px" }, style={ 'marginTop': '3rem' }
                       )

    ],
    id="sidebar",
    style=SIDEBAR_STYLE,
)
# content design
content = html.Div ([
    dbc.Button (">>", outline=True, color="secondary", className="mr-1", id="btn_sidebar2",
                style={ "position": "fixed", "margin-left": "5px", 'visibility': 'hidden' }),
    html.P (
        "Ceyhun SAHIN: Projet 7 'Prêt à dépenser' / Formation OpenClassRooms Data Scientist",
        style={ 'margin': '1rem', 'font-size': '50px', 'font-family': 'Arial, Helvetica, sans-serif',
                'font_weight': 'bold' }
    ),
    html.P (
        "Informations générales clients (index = ID de la demande de prêt):",
        style={ 'margin': '1rem', 'font-size': '30px', 'font-family': 'Arial, Helvetica, sans-serif' }
    ),
    html.P (
        "Vous pouvez ajouter ou enlever une donnée présente dans cette liste:",
        style={ 'margin': '1rem', 'font-size': '20px', 'font-family': 'Arial, Helvetica, sans-serif' },
    ),
    dcc.Dropdown (id='features',
                  options=[{ 'label': i, 'value': i } for i in
                           df_test.columns],
                  multi=True,
                  style={ 'width': '100%' },

                  clearable=True,
                  placeholder='Features',
                  value=df_test.columns[:7]
                  ),
    dash_table.DataTable (
        id='datatable-interactivity',
        columns=[
            { "name": i, "id": i, "deletable": True, "selectable": True } for i in df_test.columns[:7]
        ],
        data=df_test.to_dict ('records'),
        editable=True,
        # style_as_list_view=True,
        filter_action="native",
        sort_action="native",
        sort_mode="multi",
        column_selectable="single",
        row_selectable="multi",
        row_deletable=True,
        selected_columns=[],
        selected_rows=[],
        page_action="native",
        page_current=0,
        page_size=10,
        style_table={
            'height': 400, 'width': 1200, 'overflowY': 'scroll'
        },
        style_data={
            'width': '150px', 'minWidth': '150px', 'maxWidth': '150px',
            'overflow': 'hidden',
            'textOverflow': 'ellipsis',
        },
        style_cell_conditional=[
            {
                'if': { 'column_id': c },
                'textAlign': 'left'
            } for c in df_test.columns
        ]

    ),
    html.Div (id='datatable-interactivity-container'),
    html.Div (id='hidden_data', children = [] , style = {'visibility': 'hidden'}),

],

    id="page-content",
    style=CONTENT_STYLE
)
# content collapse design
collapse = html.Div (
    [
        dbc.Button (
            "Informations Complémentaire",
            id="collapse-button",
            className="mb-3",
            color="primary",
            n_clicks=0,
        ),
        dbc.Collapse (
            dbc.Card (dbc.CardBody ("Ici vous trouvez les informations disponibles pour tous les clients.\ Pour plus "
                                    "d'informations sur les features (variables) disponibles merci de contacter "
                                    "l'équipe support.")),
            id="collapse",
            is_open=False,
        ),
    ], style=CONTENT_STYLE, id='collapse_id'
)

# client content design
client_content = html.Div (id="page-content_client", children=[
    html.P (
        "Données du client, demande",
        style={ 'margin': '1rem', 'font-size': '50px', 'font-family': 'Arial, Helvetica, sans-serif',
                'font_weight': 'bold' },
        id='client_id'),

    dcc.Dropdown (id='features_client',
                  options=[{ 'label': i, 'value': i } for i in
                           df_test.columns],
                  multi=True,
                  style={ 'width': '100%' },

                  clearable=True,
                  placeholder='Features',
                  value=df_test.columns[1:7]
                  ),
    dash_table.DataTable (
        id='datatable-interactivity_client',
        columns=[
            { "name": i, "id": i, "deletable": True, "selectable": True } for i in df_test.columns[1:7]
        ],
        data=df_test.to_dict ('records'),
        editable=True,
        # style_as_list_view=True,
        filter_action="native",
        sort_action="native",
        sort_mode="multi",
        column_selectable="single",
        row_selectable="multi",
        row_deletable=True,
        selected_columns=[],
        selected_rows=[],
        page_action="native",
        page_current=0,
        page_size=10,
        style_table={
            'height': 150, 'width': 1200, 'overflowY': 'scroll'
        },
        style_data={
            'width': '150px', 'minWidth': '150px', 'maxWidth': '150px',
            'overflow': 'hidden',
            'textOverflow': 'ellipsis',
        },
        style_cell_conditional=[
            {
                'if': { 'column_id': c },
                'textAlign': 'left'
            } for c in df_test.columns
        ],

    ),
    html.Div (id='datatable-interactivity-container_client'),

    dcc.Loading (id='demande_graph_loading2', type='cube',
                     children=[html.Div (
                         dcc.Graph (id='graph_client_waterfall', config=config,style={'height' : 700},
                                    figure={
                                        'layout': { 'legend': { 'tracegroupgap': 0 },

                                                    }
                                    }
                                    ))]),

],

                           style=CONTENT_STYLE_client
                           )

# Clietn content collapse design
collapse2 = html.Div (
    [
        dbc.Button (
            "Informations Complémentaire",
            id="collapse-button2",
            className="mb-3",
            color="primary",
            n_clicks=0,
        ),
        dbc.Collapse (
            dbc.Card (dbc.CardBody ("Ici vous trouvez les informations client disponibles pour la demande de prêt "
                                    "sélectionnée.\ La graphique en bâton donne les valeurs de features (variables) \ "
                                    "normalisées pour pouvoir les afficher sur la même échelle.")),
            id="collapse2",
            is_open=False,

        ),
    ], style=CONTENT_STYLE_client, id='collapse_id2'
)

# resultats de demande design
resultat_de_demande = html.Div ([
    html.P (
        "Décision sur la demande de prêt",
        style={ 'margin': '1rem', 'font-size': '30px', 'font-family': 'Arial, Helvetica, sans-serif',
                'font_weight': 'bold' }
    ),
    html.P ('',
            style={ 'margin': '1rem', 'font-size': '20px', 'font-family': 'Arial, Helvetica, sans-serif' },
            id='décision1'),
    html.P ("",
            style={ 'margin': '1rem', 'font-size': '20px', 'font-family': 'Arial, Helvetica, sans-serif' },
            id='décision2'),
    html.P ("",
            style={ 'margin': '1rem', 'font-size': '20px', 'font-family': 'Arial, Helvetica, sans-serif' },
            id='décision3'),

    html.Div (children=[daq.Gauge (
        color={ "gradient": True, "ranges": { "green": [0, 50], "yellow": [50, 60], "red": [60, 100] } },
        value=0,
        label='Score (%)',
        max=100,
        min=0,
        size=300,
        id='gauge1'
    ),

        daq.LEDDisplay (
            label="Score (%)",
            value="",
            size=100,
            id='led1'
        )], style={ 'display': 'flex', 'flex-direction': 'row', 'justify-content': 'space-evenly' }),

    html.Div (children=
              [dcc.Loading (id='shap_graph_loading', type='graph',
                            children=[html.H3 ('Importance Globale'),
                                      html.Div([
                                            html.Div (id='shap_graph_total'),
                                            html.Div (id='shap_graph_total_2')],
                                            style={ 'display': 'flex', 'flex-direction': 'row',
                                                    'justify-content': 'space-evenly' }

                                      )]),
               dcc.Loading (id='imp_graph_loading', type='graph',
                            children=[html.Div (dcc.Graph (id='shap_graph_imp', config=config))])]),
    #html.Div(id = 'explainer', children = [])
],

    id="page_resultat",
    style=CONTENT_STYLE_client
)

# resultats de demande collapse design
collapse3 = html.Div (
    [
        dbc.Button (
            "Informations Complémentaire",
            id="collapse-button3",
            className="mb-3",
            color="primary",
            n_clicks=0,
        ),
        dbc.Collapse (
            dbc.Card (dbc.CardBody ([
                html.H2 (
                    "Le retour de l'API de prédiction donne un score entre 0 et 100% qui représente la probabilité de "
                    "refus de prêt."),
                html.H3 ("Trois cas de figure sont alors possibles:"),
                html.P ("1) Le score est en dessous de 50% → la demande de prêt est acceptée."),
                html.P (
                    "2) Le score est entre 50 et 60% → la demande de prêt est refusée mais peut être discutée avec le "
                    "conseiller"),
                html.P ("3) Le score est au dessus de 61% → la demande de prêt est refusée..")])),
            id="collapse3",
            is_open=False,
        ),
    ], style=CONTENT_STYLE_client, id='collapse_id3'
)

# Troisième collapse design
collapse4 = html.Div (
    [
        dbc.Button (
            "Informations Complémentaire",
            id="collapse-button4",
            className="mb-3",
            color="primary",
            n_clicks=0,
        ),
        dbc.Collapse (
            dbc.Card (dbc.CardBody ([
                html.P (
                    "Ce graphique permet d'afficher un nuage de points en fonction de deux features sélectionnables."),
                html.P (" Le code couleur indique la valeur du score client.")])),
            id="collapse4",
            is_open=False,
        ),
    ], style=CONTENT_STYLE_client, id='collapse_id4'
)

# Troisième collapse design
collapse5 = html.Div (
    [
        dbc.Button (
            "Informations Complémentaire",
            id="collapse-button5",
            className="mb-3",
            color="primary",
            n_clicks=0,
        ),
        dbc.Collapse (
            dbc.Card (dbc.CardBody ([
                html.P (
                    "Pour Histogramme : Histogramme permet d'afficher les distributions des groupes de clients."),
                html.P ("Pour Boxplot : Boxplot permet d\'afficher les distributions des groupes de clients en fonction de la valeur du client selectionne. \n"
                        " Notez que les variables sont normalisees afin d\'avoir une image de la situation de notre client \n"
                        " par rapport aux autres groupes de clients.")])),
            id="collapse5",
            is_open=False,
        ),
    ], style=CONTENT_STYLE_client, id='collapse_id5'
)
# analise de client et son explication par rapport aux 2 features
client_analyse = html.Div ([
    html.H2 ([
        html.I ("Analyse bivariée")],
        style={ 'margin': '1rem', 'font-size': '30px', 'font-family': 'Arial, Helvetica, sans-serif',
                'font_weight': 'bold' }
    ),

    html.Div (children=[dcc.Dropdown (id='first_par',
                                      options=[{ 'label': i, 'value': i } for i in
                                               df_test.columns],
                                      multi=False,
                                      style={ 'width': '100%' },

                                      clearable=True,
                                      placeholder='Features',
                                      value='AMT_INCOME_TOTAL'
                                      ),

                        dcc.Dropdown (id='second_par',
                                      options=[{ 'label': i, 'value': i } for i in
                                               df_test.columns],
                                      multi=False,
                                      style={ 'width': '100%' },
                                      clearable=True,
                                      placeholder='Features',
                                      value='AMT_CREDIT'
                                      ), ],
              style={ 'display': 'flex', 'flex-direction': 'row', 'justify-content': 'space-evenly' }),

    dcc.Loading (id='bivarie_graph_loading', type='cube',
                 children=[html.Div (dcc.Graph (id='bivarie_graph', config=config, style={'height' : 700}), )]),
    ],    id="cl_analy",
          style=CONTENT_STYLE_client)

client_analyse_uni = html.Div([
    html.H2 ([
        html.I ("Analyse univariée")],
        style={ 'margin': '1rem', 'font-size': '30px', 'font-family': 'Arial, Helvetica, sans-serif',
                'font_weight': 'bold' }
    ),
    html.P ('Vous pouvez sélectionner un graphique',
                style={ 'margin': '1rem', 'font-size': '20px', 'font-family': 'Arial, Helvetica, sans-serif' },
                ),

    html.Div (children=[dcc.Dropdown (id='type_graph',
                                      options=[{ 'label': i, 'value': i } for i in
                                               ['Boxplot', 'Batôn']],
                                      multi=False,
                                      style={ 'width': '80%', 'margin': 10},

                                      clearable=True,
                                      placeholder='Features',
                                      value='Boxplot',
                                      ),
                        ],
              ),


    html.Div (children=[dcc.Dropdown (id='uni_first_par',
                                      options=[{ 'label': i, 'value': i } for i in
                                               df_test.columns],
                                      multi=True,
                                      style={ 'width': '80%' , 'margin': 10},

                                      clearable=True,
                                      placeholder='Features',
                                      value=['AMT_INCOME_TOTAL','AMT_CREDIT']
                                      ),
                        ],
              ),



    dcc.Loading (id='univarie_graph_loading', type='cube',
                 children=[html.Div (dcc.Graph (id='univarie_graph', config=config,style={'height' : 700}), )]),
],

    id="cl_analy_uni",
    style=CONTENT_STYLE_client
)



# app layout general
#--------------------------------------->
app.layout = html.Div (
    [
        dcc.Location(id='url', refresh=False),
        html.Div(id='page-content_base'),
        dcc.Store (id='side_click'),
    ],
)

@app.callback(Output('page-content_base', 'children'),
              Input('url', 'pathname'))
def display_page(pathname):
    if pathname == '/':
        return sidebar,content,collapse,client_content,collapse2,resultat_de_demande,collapse3,\
               client_analyse,collapse4,client_analyse_uni,collapse5

#--------------------------------------->

# sidebar function for X and >> buttons
@app.callback (
    [
        Output ("sidebar", "style"),
        Output ("page-content", "style"),
        Output ("side_click", "data"),
        Output ("btn_sidebar2", "style"),
        Output ("collapse_id", "style"),
        Output ("hidden_data", "children")

    ],

    [Input ("btn_sidebar", "n_clicks"), Input ("btn_sidebar2", "n_clicks")],
    [State ("side_click", "data"),State ("hidden_data", "children")]
)
def toggle_sidebar(n, n1, nclick, hidden):
    q1 = dash.callback_context.triggered[0]["prop_id"].split (".")[0]

    if q1 == 'btn_sidebar' :
        hidden.append('btn_sidebar')

        if len(hidden) == 1:

            sidebar_style = SIDEBAR_STYLE
            content_style = CONTENT_STYLE
            cur_nclick = 'SHOW'
            return sidebar_style, content_style, cur_nclick, { 'visibility': 'hidden' }, CONTENT_STYLE, hidden
        elif len(hidden) > 1:
            sidebar_style = SIDEBAR_HIDEN
            content_style = CONTENT_STYLE1
            cur_nclick = "HIDDEN"
            return sidebar_style, content_style, cur_nclick, { 'visibility': 'visible' }, CONTENT_STYLE1, hidden
    else :
        sidebar_style = SIDEBAR_STYLE
        content_style = CONTENT_STYLE
        cur_nclick = 'SHOW'
        return sidebar_style, content_style, cur_nclick, { 'visibility': 'hidden' }, CONTENT_STYLE, hidden


# for the first datatable function
@app.callback (
    Output ('datatable-interactivity', 'style_data_conditional'),
    Input ('datatable-interactivity', 'selected_columns')
)
def update_styles(selected_columns):
    return [{
        'if': { 'column_id': i },
        'background_color': '#D2F3FF'
    } for i in selected_columns]


# les features choosen show when datatable contents change
@app.callback (
    Output ('datatable-interactivity', 'columns'),
    Input ('features', 'value')
)
def update_table(selected_col):
    return [{ "name": i, "id": i, "deletable": True, "selectable": True } for i in selected_col]


# collapse overture/fermeture function
@app.callback (
    Output ("collapse", "is_open"),
    [Input ("collapse-button", "n_clicks")],
    [State ("collapse", "is_open")],
)
def toggle_collapse(n, is_open):
    if n:
        return not is_open
    return is_open


# collapse overture/fermeture function
@app.callback (
    Output ("collapse2", "is_open"),
    [Input ("collapse-button2", "n_clicks")],
    [State ("collapse2", "is_open")],
)
def toggle_collapse2(n, is_open):
    if n:
        return not is_open
    return is_open

# collapse overture/fermeture function
@app.callback (
    Output ("collapse3", "is_open"),
    [Input ("collapse-button3", "n_clicks")],
    [State ("collapse3", "is_open")],
)
def toggle_collapse3(n, is_open):
    if n:
        return not is_open
    return is_open

# collapse overture/fermeture function
@app.callback (
    Output ("collapse4", "is_open"),
    [Input ("collapse-button4", "n_clicks")],
    [State ("collapse4", "is_open")],
)
def toggle_collapse4(n, is_open):
    if n:
        return not is_open
    return is_open
# collapse overture/fermeture function
@app.callback (
    Output ("collapse5", "is_open"),
    [Input ("collapse-button5", "n_clicks")],
    [State ("collapse5", "is_open")],
)
def toggle_collapse5(n, is_open):
    if n:
        return not is_open
    return is_open

# when sidebar was cleaned
@app.callback (

    [Output ('page-content_client', 'style'), Output ("collapse_id2", "style")],
    [Input ('stades_client', 'value'), Input ('pret_id', 'value'),
     Input ("btn_sidebar", "n_clicks"), Input ("btn_sidebar2", "n_clicks")], )
def update_table_client_visibility(tick1, cl_id, sd1, sd2):
    if cl_id == None:
        raise PreventUpdate
    q1 = dash.callback_context.triggered[0]["value"]
    q2 = dash.callback_context.triggered[0]["prop_id"].split (".")[0]
    if cl_id != None:
        if not tick1:
            return CONTENT_STYLE_client, CONTENT_STYLE_client
        if 'don_client' not in tick1:
            return CONTENT_STYLE_client, CONTENT_STYLE1_client
        if q2 == 'btn_sidebar':
            return CONTENT_STYLE1_client, CONTENT_STYLE1_client
        marge = { "margin-left": "30rem", 'visibility': 'visible' }
        return marge, marge

@app.callback (

    [Output ('page_resultat', 'style'), Output ("collapse_id3", "style")],
    [Input ('stades_client', 'value'), Input ('pret_id', 'value'),
     Input ("btn_sidebar", "n_clicks"), Input ("btn_sidebar2", "n_clicks")], )
def update_demo_visibility(tick1, cl_id, sd1, sd2):
    if cl_id == None:
        raise PreventUpdate
    q2 = dash.callback_context.triggered[0]["prop_id"].split (".")[0]
    if cl_id != None:
        if not tick1:
            return CONTENT_STYLE_client, CONTENT_STYLE_client

        if 'result_dem' not in tick1:
            return CONTENT_STYLE_client, CONTENT_STYLE_client
        if q2 == 'btn_sidebar':
            return CONTENT_STYLE1_client, CONTENT_STYLE1_client
        marge = { "margin-left": "30rem", 'visibility': 'visible' }
        return marge, marge
    else : return no_update


@app.callback (

    [Output ('cl_analy', 'style'), Output ("collapse_id4", "style")],
    [Input ('stades_client', 'value'), Input ('pret_id', 'value'),
     Input ("btn_sidebar", "n_clicks"), Input ("btn_sidebar2", "n_clicks")], )
def update_analyse_visibility(tick1, cl_id, sd1, sd2):
    if cl_id == None:
        raise PreventUpdate
    q2 = dash.callback_context.triggered[0]["prop_id"].split (".")[0]
    if cl_id != None:
        if not tick1:
            return CONTENT_STYLE_client, CONTENT_STYLE_client

        if 'analyse_client' not in tick1:
            return CONTENT_STYLE_client, CONTENT_STYLE_client
        if q2 == 'btn_sidebar':
            return CONTENT_STYLE1_client, CONTENT_STYLE1_client
        marge = { "margin-left": "30rem", 'visibility': 'visible' }
        return marge, marge
    else:
        return no_update

@app.callback (

    [Output ('cl_analy_uni', 'style'), Output ("collapse_id5", "style")],
    [Input ('stades_client', 'value'), Input ('pret_id', 'value'),
     Input ("btn_sidebar", "n_clicks"), Input ("btn_sidebar2", "n_clicks")], )
def update_analyse_visibility_collapse5(tick1, cl_id, sd1, sd2):
    if cl_id == None:
        raise PreventUpdate
    q2 = dash.callback_context.triggered[0]["prop_id"].split (".")[0]
    if cl_id != None:
        if not tick1:
            return CONTENT_STYLE_client, CONTENT_STYLE_client

        if 'analyse_client' not in tick1:
            return CONTENT_STYLE_client, CONTENT_STYLE_client
        if q2 == 'btn_sidebar':
            return CONTENT_STYLE1_client, CONTENT_STYLE1_client
        marge = { "margin-left": "30rem", 'visibility': 'visible' }
        return marge, marge
    else:
        return no_update

# add features to datatable
@app.callback (
    [Output ('client_id', 'children'),
     Output ('datatable-interactivity_client', 'data'),
     Output ('datatable-interactivity_client', 'columns')],
    [Input ('pret_id', 'value'), Input ('features_client', 'value')]
)
def update_table_client(client_id, sel_col):
    if client_id == None:
        raise PreventUpdate
    value = f"Données du client, demande '{client_id}'"
    df_client = df_test[df_test['SK_ID_CURR'] == client_id]

    return value, df_client.to_dict ('records'), [{ "name": i, "id": i, "deletable": True, "selectable": True } for i in
                                                  sel_col]


# add waterfall graph
@app.callback (Output ('graph_client_waterfall', 'figure'),
               Input ('pret_id', 'value'),
               Input ('features_client', 'value'))
def graph_client(cl_id, feat_state):
    if cl_id == None:
        raise PreventUpdate
    df_client = df_test_normalize[df_test_normalize.index == cl_id]
    time.sleep (1)

    fig = go.Figure (go.Waterfall (
        name=f"Client : {cl_id}",
        orientation="v",
        measure= ['relative' for i in range(len(feat_state))],
        x=df_client[feat_state].columns.tolist (),
        textposition="outside",
        #text=["+60", "+80", "", "-40", "-20", "Total"],
        y=df_client[feat_state].values.tolist ()[0],
        connector={ "line": { "color": "rgb(63, 63, 63)" } },
    ))

    fig.update_layout (
        title=f"Diagramme bar données ID: {cl_id}",
        xaxis=dict (title='Features'), yaxis=dict (title='Valeurs Normalisée'),
        showlegend=True
    )

    return fig


# La fonction retourne score via API, graphics

@app.callback (
    [Output ('décision1', 'children'), Output ('décision2', 'children'),
     Output ('décision3', 'children'),
     Output ('gauge1', 'value'), Output ('led1', 'value'),
     Output ('décision2', 'style'), Output ('décision3', 'style'),
     Output ('led1', 'color'), Output ('shap_graph_imp', 'figure'),
     Output ('shap_graph_total', 'children'),Output ('shap_graph_total_2', 'children') ],
    Input ('stades_client', 'options'), Input ('pret_id', 'value'),
)
def result_client(feat_cl, client_id):  # sourcery no-metrics
    if client_id == None:
        raise PreventUpdate


    if 'result_dem' not in [i['value'] for i in feat_cl]:
        return no_update

    url_api_model_result = 'https://oc-p7-home-risk-flaskapi.herokuapp.com/scores'
    get_request = requests.get (url=url_api_model_result, params={ 'index': client_id })
    get_request.raise_for_status ()
    score, data = '', ''
    if get_request.status_code != 204:
        score = get_request.json ()['Credit_score']
        data = get_request.json ()['json_data']

    score *= 100
    score = round (score, 2)

    value1 = f"Demande de prêt ID: '{client_id}'"
    value2 = f"Probabilité de défaut de remboursement: {score:,.2f}%"
    if score > 61:
        value3 = 'Demande de prêt réfusé'
        color = { 'color': 'red' }
        col = 'red'
    elif 50 <= score <= 60:
        value3 = 'il faut discuter avec le conseiller'
        color = { 'color': 'grey' }
        col = 'grey'
    else:
        value3 = 'Demande de prêt Accéptée'
        color = { 'color': 'green' }
        col = 'green'

    shap.initjs ()
    df_client = pd.read_json (data)
    clf = model
    explainer = shap.TreeExplainer (clf, data=None, model_output='raw', feature_perturbation='tree_path_dependant')
    shap_values = explainer.shap_values (df_client)
    feature_imp = pd.DataFrame (sorted (zip (shap_values[0], df_client.columns)
                                        ), columns=['Value', 'Feature'])
    data = feature_imp.sort_values (by="Value", ascending=False)[:20]

    shap_html_1 = html.Img (src='data:image/png;base64,{}'.format (encoded_image_1.decode ()), height=600, width=600)
    shap_html_2 = html.Img (src='data:image/png;base64,{}'.format (encoded_image_2.decode ()), height=600, width=600)

    fill_color = ["green" if (val >= 0) & (col == 'green') else "grey " if (val >= 0) & (col == 'grey') else "red"
                  for val in data.Value]
    time.sleep (1)
    fig = go.Figure ()
    fig.add_trace (
        go.Bar (
            x=data.Value,
            y=data.Feature,
            orientation="h",
            marker_color=fill_color,
        )
    )
    fig.update_traces (textposition="outside")
    fig.update_layout (
        yaxis=dict (autorange="reversed", title='Features'),
        height=600,
        template="simple_white",
        title_text=f"Importance Locale de la Client <<{client_id}>>",
        title_font_size= 24,

        xaxis=dict (title='Average Impact on Model Output Magnitude',)
    )

    return value1, value2, value3, score, score, color, color, col, fig, shap_html_1,shap_html_2

# add bivarie graph side
@app.callback (
    Output ('bivarie_graph', 'figure'),
    [Input ('first_par', 'value'), Input ('second_par', 'value')],
    [Input ('stades_client', 'options'), Input ('pret_id', 'value')],
)
def result_client2(f1, f2, feat_cl, client_id):
    if feat_cl == None:
        raise PreventUpdate

    ctx = dash.callback_context
    input_id = ctx.triggered[0]["prop_id"].split (".")[0]
    if 'analyse_client' in [i['value'] for i in feat_cl]:

        total_score = query_data ()

        fig2 = go.Figure ()
        # all client scatter filtered with Predict column and treshold (accepted / denied) ####
        fig2.add_trace (go.Scatter (x=total_score[total_score['Predict'] == 0][f1],
                                    y=total_score[total_score['Predict'] == 0][f2],
                                    mode='markers', name='Tous les clients_prêt_acceptés', marker_symbol='diamond',
                                    marker={ 'size': 10,
                                             'color': total_score.loc[total_score['Predict'] == 0]['Predict'],
                                             'coloraxis': 'coloraxis' }))
        fig2.add_trace (go.Scatter (x=total_score[total_score['Predict'] == 1][f1],
                                    y=total_score[total_score['Predict'] == 1][f2],
                                    mode='markers', name='Tous les clients_prêt_refusés', marker_symbol='x',
                                    marker={ 'size': 10, 'color': total_score[total_score['Predict'] == 1]['Predict'],
                                             'coloraxis': 'coloraxis' }))

        # plot selected client point ####
        total_score.index = total_score.index.map (float)

        fig2.add_trace (
            go.Scatter (x=total_score[total_score.index == client_id][f1],
                        y=total_score[total_score.index == client_id][f2],
                        mode='markers', name='ID_prêt_client_selectionné',
                        marker={ 'size': 15, 'color': total_score[total_score.index == client_id]['Predict'],
                                 'coloraxis': 'coloraxis',
                                 'line': { 'width': 3, 'color': 'black' } }))
        # update legend localisation and add colorbar ####
        fig2.update_layout (legend={ 'orientation': "h", 'yanchor': 'bottom', 'y': 1.05, 'xanchor': 'right', 'x': 1,
                                     'bgcolor': 'White' },
                            xaxis={ 'title': f1 }, yaxis={ 'title': f2 },
                            coloraxis={ 'colorbar': { 'title': 'Score' },
                                        'colorscale': 'viridis', 'cmin': 0, 'cmax': 1, 'showscale': True })

        return fig2
    else:
        return no_update

# add univarie graph side
@app.callback (
    Output ('uni_first_par', 'multi'),
    Output ('uni_first_par', 'value'),
    Input ('type_graph', 'value'),
)
def select_graph(type_gr):
    if type_gr == 'Boxplot' :
        return True, ['AMT_INCOME_TOTAL','AMT_CREDIT']
    else: return False, 'AMT_INCOME_TOTAL'


@app.callback (
    Output ('univarie_graph', 'figure'),
    [Input ('uni_first_par', 'value'),Input ('stades_client', 'options'),
     Input ('pret_id', 'value'),Input ('type_graph', 'value')],
)
def univarie_graph(uni_f1, feat_cl, client_id, type_gr):
    if feat_cl == None or client_id == None:
        raise PreventUpdate

    if 'analyse_client' in [i['value'] for i in feat_cl]:

        total_score = query_data ()
        fig3 = go.Figure ()
        #fig4 = go.Figure ()
        if type_gr == 'Boxplot':

            for idx, col in enumerate (total_score[uni_f1].columns, 0):

                val = total_score[total_score.index == str (client_id)]['Predict'].values

                df_plot_acc1 = total_score[total_score['Predict'] == 0]
                df_plot_acc2 = total_score[total_score['Predict'] == 1]
                situation = ''
                if val[0] == 0 :
                    situation = 'Refusée'

                    fig3.add_trace (
                        go.Scatter (x=[col], y=df_plot_acc1[df_plot_acc1.index == str (client_id)][col],
                                    name=f"{client_id} : Acceptée"))
                    fig3.add_trace (go.Box (y=df_plot_acc1[col], name=f"{col}"))
                    fig3.update_layout (boxmode='overlay', boxgroupgap=0.1)
                    fig3.add_trace (go.Box (y=df_plot_acc2[col], name=f"{col}: {situation}"))

                else :
                    situation = 'Acceptée'
                    fig3.add_trace (
                        go.Scatter (x=[col], y=df_plot_acc2[df_plot_acc2.index == str (client_id)][col],
                                    name=f"{client_id} : Refusée"))
                    fig3.add_trace (go.Box (y=df_plot_acc2[col], name=f"{col}"))
                    fig3.update_layout (boxmode='overlay', boxgroupgap=0.1)
                    fig3.add_trace (go.Box (y=df_plot_acc1[col], name=f"{col} : {situation}"))

            fig3.update_layout (boxmode='overlay', xaxis_tickangle=0,
                                title='Tous les clients par rapport au acceptées et refusées',
                                yaxis_title='Valeurs Normalisée', xaxis_title='Features')

            return fig3

        else :
            total_score['Acc_Ref'] = np.where(total_score['Predict']==0, 'Acceptée','Refusée')
            fig4 = px.histogram(total_score, x=uni_f1, color="Acc_Ref", marginal="rug",
                                color_discrete_sequence = ['blue','red'],
                                )

            fig4.add_trace (go.Histogram (x=total_score[total_score.index == str (client_id)][uni_f1],

                                        name=f"{uni_f1} : {client_id}",marker  = { 'color' : '#330C73' }))

            fig3.update_layout (title='Tous les clients ')


            return fig4

    else:
        return no_update


if __name__ == '__main__':
    app.run_server (port=8050, debug=False)
