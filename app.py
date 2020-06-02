# Import required libraries
import pickle
import copy
import pathlib
import dash
import math
import sklearn
import datetime as dt
import pandas as pd
import numpy as np
from dash.dependencies import Input, Output, State, ClientsideFunction
import dash_core_components as dcc
import dash_html_components as html


# get relative data folder
PATH = pathlib.Path(__file__).parent
DATA_PATH = PATH.joinpath("data").resolve()

app = dash.Dash(
    __name__, meta_tags=[{"name": "viewport", "content": "width=device-width"}]
)
server = app.server



# Load data
data = pd.read_csv(DATA_PATH.joinpath("data_v2_raw.csv"), low_memory=False)
with open(DATA_PATH.joinpath("Random_Forest_Model.pkl"), 'rb') as file:  
    model = pickle.load(file)

# Function
ref_dictionary = {
    'Large Company': 0,
    'Medium Company': 1,
    'Small Company': 2,
    'Action': 0,
    'Action-Adventure': 1,
    'Adventure': 2,
    'Fighting': 3,
    'MMO': 4,
    'Misc': 5,
    'Music': 6,
    'Party': 7,
    'Platform': 8,
    'Puzzle': 9,
    'Racing': 10,
    'Role-Playing': 11,
    'Shooter': 12,
    'Simulation': 13,
    'Sports': 14,
    'Strategy': 15,
    'Visual Novel': 16,
    'Level1': 0, 
    'Level2': 2, 
    'Level3': 3, 
    'Level4': 1,
    'Game Console': 0, 
    'Handheld': 1, 
    'PC': 2
}

def get_input_arr(input_dict):
  input_arr = []
  if_not_score = 1 if input_dict['Critic_Score'] == 0.0 else 0
  show_all_company = 1 if input_dict['Company'] == "Show All" else 0
  company_size_list = [0] * 3

  rating_list = [0] * 4
  idx = ref_dictionary[input_dict['Rating']]
  rating_list[idx] = 1

  platform_list = [0] * 3
  idx = ref_dictionary[input_dict['Platform']]
  platform_list[idx] = 1

  genre_list = [0] * 17
  idx = ref_dictionary[input_dict['Genre']]
  genre_list[idx] = 1

  if not show_all_company:
    idx = ref_dictionary[input_dict['Company']]
    company_size_list[idx] = 1
    input_arr = np.asarray([input_dict['Critic_Score'], np.log(input_dict['Price']), if_not_score] + company_size_list + rating_list + platform_list + genre_list).reshape(1, -1)
  else:
    for i in range(3):
      company_size_list = [0] * 3
      company_size_list[i] = 1
      cur_arr = [input_dict['Critic_Score'], np.log(input_dict['Price']), if_not_score] + company_size_list + rating_list + platform_list + genre_list
      input_arr.append(cur_arr) 
    input_arr = np.asarray(input_arr)
  return input_arr


# Create global chart template
mapbox_access_token = "pk.eyJ1IjoicGxvdGx5bWFwYm94IiwiYSI6ImNrOWJqb2F4djBnMjEzbG50amg0dnJieG4ifQ.Zme1-Uzoi75IaFbieBDl3A"

layout = dict(
    autosize=True,
    automargin=True,
    margin=dict(l=30, r=30, b=20, t=40),
    hovermode="closest",
    plot_bgcolor="#F9F9F9",
    paper_bgcolor="#F9F9F9",
    legend=dict(font=dict(size=10), orientation="h"),
    title="Satellite Overview",
    mapbox=dict(
        accesstoken=mapbox_access_token,
        style="light",
        center=dict(lon=-78.05, lat=42.54),
        zoom=7,
    ),
)

# Create app layout
app.layout = html.Div(
    [
        dcc.Store(id="aggregate_data"),
        # empty Div to trigger javascript file for graph resizing
        html.Div(id="output-clientside"),
        html.Div(
            [
                html.Div(
                    [
                        html.Div(
                            [
                                html.H1(
                                    "Video Game Sales Prediction",
                                    style={"text-align":"center"},
                                ),
                            ]
                        )
                    ],
                    className="one-half column",
                    id="title",
                ),
            ],
            id="header",
            className="row flex-display",
            style={"margin-bottom": "25px"},
        ),


        html.Div(
            [
                html.Div(
                    [
                        html.P("Choose the sealed price: ", className="control_label"),
                        dcc.Slider(
                            id="price_slider",
                            min=0.0,
                            max=500.0,
                            step=0.01,
                            value=19.99
                        ),
                        html.P(id="price_text"),

                        html.P("Choose the critic score: ", className="control_label"),
                        dcc.Slider(
                            id="critic_score",
                            min=0.0,
                            max=10.0,
                            step=0.01,
                            value=0.0,
                        ),
                        html.P(id="score_text"),

                        html.P("Choose the company size of the publisher: "),
                        dcc.Dropdown(
                            id="company_size",
                            options=[
                                {'label': 'Large Company(Published > 100 games)', 'value': 'Large Company'},
                                {'label': 'Medium Company(Published > 50 games)', 'value': 'Medium Company'},
                                {'label': 'Small Company(Published <= 50 games)', 'value': 'Small Company'},
                                {'label': 'Show All Possible Results', 'value': 'Show All'},
                            ],
                            placeholder = "Select a Company size",
                        ),
                        html.P(id="company_text")
                    ],
                    style = {'width': '49%', 'float': 'left', 'display': 'inline-block'}
                ),

                html.Div(
                    [
                        html.P("Choose the genre of your game: "),
                        dcc.Dropdown(
                            id="genre",
                            options=[
                                {'label': 'Action', 'value': 'Action'},
                                {'label': 'Action-Adventure', 'value': 'Action-Adventure'},
                                {'label': 'Adventure', 'value': 'Adventure'},
                                {'label': 'Fighting', 'value': 'Fighting'},
                                {'label': 'MMO', 'value': 'MMO'},
                                {'label': 'Misc', 'value': 'Misc'},
                                {'label': 'Music', 'value': 'Music'},
                                {'label': 'Party', 'value': 'Party'},
                                {'label': 'Platform', 'value': 'Platform'},
                                {'label': 'Puzzle', 'value': 'Puzzle'},
                                {'label': 'Racing', 'value': 'Racing'},
                                {'label': 'Role-Playing', 'value': 'Role-Playing'},
                                {'label': 'Shooter', 'value': 'Shooter'},
                                {'label': 'Simulation', 'value': 'Simulation'},
                                {'label': 'Sports', 'value': 'Sports'},
                                {'label': 'Strategy', 'value': 'Strategy'},
                                {'label': 'Visual Novel', 'value': 'Visual Novel'},
                            ],
                            placeholder = "Select a Genre",
                        ),
                        html.P(id="genre_text"),

                        html.P("Choose the ESRB rating of your game: "),
                        dcc.Dropdown(
                            id="rating",
                            options=[
                                {'label': 'Everyone(E, EC, E10)', 'value': 'Level1'},
                                {'label': 'Teenager', 'value': 'Level2'},
                                {'label': 'M', 'value': 'Level3'},
                                {'label': 'Pending/Unknown', 'value': 'Level4'},
                            ],
                            placeholder = "Select a Rating",
                        ),
                        html.P(id="rating_text"),

                        html.P("Choose the platform type you want to release your game: "),
                        dcc.Dropdown(
                            id="platform",
                            options=[
                                {'label': 'PC', 'value': 'PC'},
                                {'label': 'Handheld(PSV, NS)', 'value': 'Handheld'},
                                {'label': 'Console(PS3, PS4, XBOX)', 'value': 'Game Console'}
                            ],
                            placeholder = "Select a Platform",
                        ),
                        html.P(id="platform_text")

                    ],
                style = {'width': '49%', 'float': 'right', 'display': 'inline-block'}
                )
            ]),

        html.Div(
            [
                html.Button(id="submit_btn", n_clicks=0, children="Show Results",
                                    style = {'color': 'white','border': 'none','width': '25%', 'border-radius': '8px',
                             'background-color': '#4da6ff','text-align': 'center',
                             'text-decoration': 'none','display':'block','margin':'0 auto',
                                             'padding': '14px 40px'}),

            ],
            style = {'width': '100%', 'position': '','margin':'0 auto','display': 'inline-block','align-items': 'center'}

            ),

        html.Div(
            [
                html.Div(
                    [dcc.Graph(id="sales_graph")],
                    className="pretty_container",
                ),
            ],
            id="right-column",
            className="pretty_container eight columns",
        ),


    ],
    style = {'width': '90%','margin':'0 auto'}
)

# Create callbacks
# app.clientside_callback(
#     ClientsideFunction(namespace="clientside", function_name="resize"),
#     Output("output-clientside", "children"),
#     [Input("sales_graph", "figure")],
# )

# Print the text of price slider
@app.callback(Output("price_text", "children"), [Input("price_slider", "value")])
def update_price(value):
    return 'You have selected "{}"'.format(value)

# Print the text of critic score slider
@app.callback(Output("score_text", "children"), [Input("critic_score", "value")])
def update_score(value):
    return 'You have selected "{}"'.format(value)  

# # Print the text of platform
# @app.callback(Output("platform_text", "children"), [Input("platform", "value")])
# def update_platform(value):
#     return 'You have selected "{}"'.format(value)

# # Print the text of genre
# @app.callback(Output("genre_text", "children"), [Input("genre", "value")])
# def update_genre(value):
#     return 'You have selected "{}"'.format(value)

# # Print the text of rating
# @app.callback(Output("rating_text", "children"), [Input("rating", "value")])
# def update_rating(value):
#     return 'You have selected "{}"'.format(value)

# # Print the text of company size
# @app.callback(Output("company_text", "children"), [Input("company_size", "value")])
# def update_company_size(value):
#     return 'You have selected "{}"'.format(value)

# Draw the graph
@app.callback(Output("sales_graph", "figure"), 
    [Input("submit_btn", "n_clicks")],
    [State("platform", "value"), 
    State("genre", "value"),
    State("rating", "value"),
    State("company_size", "value"),
    State("price_slider", "value"),
    State("critic_score", "value")
    ])
def plot_genre_figure(n_clicks, platform_value, genre_value, rating_value, company_value, price_value, score_value):
    input_dict = {
        'Company': company_value,
        "Critic_Score": score_value,
        "Genre": genre_value,
        "Platform": platform_value,
        "Price": price_value,
        "Rating":rating_value,
    }
    select_data = data[data["Genre"] == genre_value]
    df = []
    company_list = ["Large Company", "Medium Company", "Small Company"]

    # Only select part data to make histogram
    if company_value and genre_value and platform_value and rating_value:
        input_arr = get_input_arr(input_dict)
        sales = np.exp(model.predict(input_arr))
        select_data = select_data[select_data["Global_Sales"] < max(sales) + 1.5]
        select_data = select_data[select_data["Global_Sales"] > min(sales) - 1.5]
        for idx, sale in enumerate(sales):
            print("Inside if statement", sale)
            if len(sales) > 1:
                name = "prediction_{}".format(company_list[idx])
                label = company_list[idx]
            else:
                name = "prediction"
                label = "result"
            show_sale = round(sale, 2)
            df.append({
                'x': [sale],
                'y': [0],
                'mode': 'markers',
                'marker': dict(size=12, line=dict(width=2, color='DarkSlateGrey')),
                'name': name,
                "hovertemplate": "{}<br>{}<extra></extra>".format(label, show_sale)
            })

    df.append({
        'x': select_data["Global_Sales"],
        'type':'histogram',
        'autobinx': True, 
        'name': 'Global Sales',
        'marker': {'color': "#88b7e3" }
    })

    return {'data': df, 'layout': {
        'title': '1st Year Global Sales of {} game'.format(genre_value),
        'xaxis': {"title": 'Global Sales (million)'},
        'yaxis': {"title": 'Count'}
        }}


# Main
if __name__ == "__main__":
    app.run_server(debug=True)
