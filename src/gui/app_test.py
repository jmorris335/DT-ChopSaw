"""
| File: app.py 
| Info: Application interface for webapp, using Dash | Plotly
| Author: John Morris, jhmrrs@clemson.edu  
| Organization: Product Lifecycle Management Center at Clemson University, plmcenter@clemson.edu  
| Permission: Copyright (C) 2023, John Morris. All rights reserved. Should not be reproduced, edited, sourced, or utilized without written permission from the author or organization

| Version History:
| - 0.0, ?? May 2024: Initialized
"""

from dash import Dash, dcc, html, Input, Output, State, Patch, no_update
from dash.exceptions import PreventUpdate
import plotly.graph_objects as go
import logging as log
import base64

if __name__ == '__main__':
    import os, sys
    os.chdir("/Users/john_morris/Documents/Clemson/Research/DTs-Public/DT-ChopSaw")
    sys.path.insert(0, '')

from src.db.actor import DBActor
from src.db.db_names import *
from src.gui.skeleton import plotSaw
from src.auxiliary.support import gatherSawParameters

log.basicConfig(level=log.DEBUG)

PLOT_DELAY = 3 #seconds
MAX_HISTORICAL_SEQUENCES = 100
LIVE_FEED1_PATH = "src/gui/assets/live_feed_cam0.png"
LIVE_FEED2_PATH = "src/gui/assets/live_feed_cam1.png"
app = Dash(update_title=None)
db = DBActor()

def getParameters()-> dict:
    """Returns a dict of all the saw parameters and parameter keys."""
    params = gatherSawParameters()
    param_keys = getParameterKeys()
    params['param_ids'] = param_keys #Store the DB key for each parameter
    for label in param_keys.values():
        params[label] = 0.0 #Set default values for parameters
    return params

def getParameterKeys()-> dict:
    """Returns a dictionary where key, value pairs are the id and label for the 
    parameters in the DB."""
    out = dict()
    param_info = db.getEntries(MOCAPMKR_TBL_NAME)
    for row in param_info:
        out[row[0]] = row[1]
    return out

app.layout = html.Div([
    html.H4('DWS780 Double Bevel Sliding Compound Miter Saw Digital Twin'),
    dcc.Store(id='store__saw_sequences', data=list()),
    dcc.Store(id='store__saw_parameters', data=getParameters()),
    dcc.Graph(id="fig__saw_3D", figure=go.Figure(plotSaw())), 
    html.Img(id='img__livefeed1', height='300px'),
    html.Img(id='img__livefeed2', height='300px'),
    dcc.Interval(id='intv__data_request', interval=400),
    dcc.Interval(id='intv__plot_update', interval=400),
    dcc.Interval(id='intv__live_feed', interval=400)
])

### Callback functionality
@app.callback(
    Output("img__livefeed1", "src"),
    Output("img__livefeed2", "src"),
    Input("intv__live_feed", "n_intervals"))
def updateLiveFeed(n_intervals):
    """Update the live feed images of the saw."""
    img1 = base64.b64encode(open(LIVE_FEED1_PATH, 'rb').read())
    img2 = base64.b64encode(open(LIVE_FEED2_PATH, 'rb').read())
    decoded_img1 = 'data:image/png;base64,{}'.format(img1.decode())
    decoded_img2 = 'data:image/png;base64,{}'.format(img2.decode()) 
    log.debug("Updated live images.")
    return decoded_img1, decoded_img2

@app.callback(
    Output("fig__saw_3D", "figure"), 
    Output("store__saw_parameters", "data"),
    Output("store__saw_sequences", "data"),
    Input("intv__plot_update", "n_intervals"),
    State("store__saw_parameters", "data"),
    State("store__saw_sequences", "data"))
def updateSawLive(n_intervals, params, sequences):
    """Update the 3D plot of the saw to the most recent sequence. Also removes 
    previous sequences."""
    if isinstance(params, list): params = params[0]
    if sequences is None or len(sequences) == 0: #no new data to plot
        log.debug("No update to saw")
        return no_update, no_update, no_update
    updateParamsFromSequence(sequences[-1], params)
    fig = getSawFigure(params)
    log.debug("Updated saw figure.")
    return fig, params, sequences

@app.callback(
    Output("store__saw_sequences", "data", allow_duplicate=True),
    Input("intv__data_request", "n_intervals"),
    State("store__saw_sequences", "data"), 
    prevent_initial_call=True )
def storeSequences(n_intervals, data: list):
    """Queries the data base for new sequences."""
    new_seq = getNewestSequence()
    if new_seq is None:
        return no_update
    log.debug(f"Added new sequence to the data store: \n\t{new_seq['params']}")
    if data is not None:
        data.append(new_seq)
        return data
    return [new_seq]


### Helper Functions
def getSawFigure(params):
    """Returns a figure of the saw with the given parameters."""
    log.debug(param2String(params))
    surfaces = plotSaw(origin=(0,0,0), 
                       bump_offset=(0., 0., 0.), 
                       miter_angle=params['miter_angle'],
                       bevel_angle=params['bevel_angle'],
                       slider_offset=params['slide_offset'],
                       crash_angle=params['crash_angle'], 
                       params=params)
    fig = go.Figure(surfaces)
    fig.update_layout(
        scene = dict(
            aspectmode='data',
            aspectratio=dict(x=1, y=1, z=1)),
        margin=dict(r=20, l=10, b=10, t=10))  
    return fig

def updateParamsFromSequence(sequence: dict, params: dict):
    """Updates the dict of parameters with the sequence values.
    
    Retrieves the label from the params dict, and then updates the value in the 
    sequence to the params dict under the mapped label.
    """
    seq_params = sequence['params']
    for p_id in seq_params.keys():
        label = params['param_ids'][p_id]
        params[label] = seq_params[p_id]

def getNewestSequence()-> dict:
    """Gets the newest seqeuence from the database."""
    db_seq = db.getLatestEntry(MOCAPSEQ_TBL_NAME)
    seq = dict(seq_id=db_seq[0], params=dict(), time=time2Float(db_seq[1]))
    params_query = f"SELECT * FROM {MOCAP_TBL_NAME} WHERE `sequence` = {seq['seq_id']}"
    db_params = db.sql(params_query)
    for row in db_params:
        seq['params'][row[3]] = row[1]
    return seq


### Utility Functions
def time2Float(time_in: str)-> float:
    """Converts the inputted time to Epoch time."""
    if isinstance(time_in, float): return time_in
    epoch = float(time_in.strftime('%s.%f'))
    return epoch

def param2String(param: dict)-> str:
    """Returns a printable string of all the parameters"""
    out = f"Miter Angle: {param['miter_angle']}\n"
    out += f"Bevel Angle: {param['bevel_angle']}\n"
    out += f"Slider Offset: {param['slide_offset']}\n"
    out += f"Crash Angle: {param['crash_angle']}\n"
    return out
        
def run():
    app.run_server(debug=False, dev_tools_silence_routes_logging=True)

if __name__ == '__main__':
    run()

"""
Seqeuence dictionary for set of coordinates taken at a single timestamp. Note
that custom classes cannot be passed to a dash Store component (even within a list).

Parameters
----------
seq_id : int
    The primary key for the sequence.
time : float
    The epoch time the sequence was observed.
params : dict, Optional
    A dict mapping a primary parameter key (int) to a float value.
"""

