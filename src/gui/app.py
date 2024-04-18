from dash import Dash, dcc, html, Input, Output, State, Patch
from dash.exceptions import PreventUpdate
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import time

PLOT_DELAY = 3 #seconds
app = Dash(__name__)

fig__saw_live = go.Figure(data=[go.Scatter3d(x=[0, 0, 1], y=[0, 0, 1], z=[0, 1, 1], mode='lines')])
fig__saw_live.update_layout(
    scene = dict(
        xaxis = dict(nticks=4, range=[0,6],),
        yaxis = dict(nticks=4, range=[0,6],),
        zaxis = dict(nticks=4, range=[0,6],),
        aspectmode='manual',
        aspectratio=dict(x=1, y=1, z=1)),
    margin=dict(r=20, l=10, b=10, t=10))

app.layout = html.Div([
    html.H4('DWS780 Double Bevel Sliding Compound Miter Saw Digital Twin'),
    dcc.Loading(
        dcc.Graph(id="graph", figure=fig__saw_live), 
        type='dot', color='#F56600'),
    dcc.Store(id='store__saw_position', data=list(
        dict( #Dummy data
            sequence=0, 
            markers=[[0,0,0], [1,1,1]], 
            time=0.0,
            is_plotted=False), 
        dict(
            sequence=1, 
            markers=[[.5,.5,.5], [1.5,1.5,1.5]], 
            time=1.0,
            is_plotted=False),
        )),
    dcc.Store(id='store__curr_sequence', data=0),
    dcc.Interval(id='data_request_intv', interval=2000),
    dcc.Interval(id='plot_update_intv', interval=250)
])

@app.callback(
    Output("graph", "extendData"), 
    Input("plot_update_intv", "n_intervals"),
    State("store__curr_sequence", "data"),
    State("store__saw_position", "data"))
def updateSawLive(n_intervals, last_plotted_seq, position_data):
    #TODO: Check if the plot needs updating by tracking the current sequence
    curr_time = time.time - PLOT_DELAY
    times = [seq['time'] for seq in position_data]
    for i, time in enumerate(times):
        if time > curr_time:
            curr_seq = i-1 if i > 0 else 0
            return getCoordinates(position_data[curr_seq])
    if len(position_data > 0):
        return getCoordinates(position_data[-1])
    raise PreventUpdate

def getCoordinates(sequence_data: dict)-> tuple:
    """Returns the `extendData` parameters with the coordinates of the given sequence."""
    markers = sequence_data['markers']
    new_x, new_y, new_z = zip(*markers)
    return dict(x=[new_x], y=[new_y], z=[new_z]), [0], len(markers)

@app.callback(
    Output("store__saw_position", "data"),
    Input("data_request_intv", "n_intervals"))
def getSawPositionData(n_intervals):
    position = dict(markers = [[0,0,0], [1,1,1]], time=[0, 0.1])
    return position

app.run_server(debug=True)