from dash import Dash, dcc, html, Input, Output, State, Patch
from dash.exceptions import PreventUpdate
import plotly.graph_objects as go
from time import time as current_time
import logging as log
import base64

from src.db.actor import DBActor
from src.db.db_names import *

log.basicConfig(level=log.DEBUG)

PLOT_DELAY = 3 #seconds
MAX_HISTORICAL_SEQUENCES = 100
LIVE_FEED1_PATH = "src/gui/assets/live_feed_cam0.png"
LIVE_FEED2_PATH = "src/gui/assets/live_feed_cam1.png"
app = Dash(update_title=None)
db = DBActor()

def getDummySequence(num_markers: int=6):
    markers = dict()
    for i in range(num_markers):
        markers[i] = [i]*3
    return dict(id=0, markers=markers, time=0.0)

fig__saw_live = go.Figure(data=[go.Scatter3d(
    x=[0, 0, 1], y=[0, 0, 1], z=[0, 1, 1], 
    mode='lines+markers',
    line= dict(width = 20))])
fig__saw_live.update_layout(
    scene = dict(
        xaxis = dict(nticks=4, range=[-150,50],),
        yaxis = dict(nticks=4, range=[0,500],),
        zaxis = dict(nticks=4, range=[-60,150],),
        aspectmode='manual',
        aspectratio=dict(x=1, y=1, z=1)),
    margin=dict(r=20, l=10, b=10, t=10))

img__live_feed1 = html.Img(
    id='img__livefeed1', 
    # src=LIVE_FEED1_PATH, 
    height='300px')

img__live_feed2 = html.Img(
    id='img__livefeed2', 
    # src=LIVE_FEED2_PATH,
    height='300px')

app.layout = html.Div([
    html.H4('DWS780 Double Bevel Sliding Compound Miter Saw Digital Twin'),
    dcc.Store(id='store__saw_position', data=[getDummySequence()]),
    # dcc.Loading(
    dcc.Graph(id="graph", figure=fig__saw_live), 
        # type='dot', color='#F56600'),
    img__live_feed1,
    img__live_feed2,
    dcc.Interval(id='data_request_intv', interval=2000),
    dcc.Interval(id='plot_update_intv', interval=400),
    dcc.Interval(id='intv__live_feed', interval=2000)
])

@app.callback(
    Output("img__livefeed1", "src"),
    Output("img__livefeed2", "src"),
    Input("intv__live_feed", "n_intervals"))
def updateLiveFeed(n_intervals):
    img1 = base64.b64encode(open(LIVE_FEED1_PATH, 'rb').read())
    img2 = base64.b64encode(open(LIVE_FEED2_PATH, 'rb').read())
    decoded_img1 = 'data:image/png;base64,{}'.format(img1.decode())
    decoded_img2 = 'data:image/png;base64,{}'.format(img2.decode()) 
    return decoded_img1, decoded_img2

@app.callback(
    Output("graph", "extendData"), 
    Output("store__saw_position", "data"),
    Input("plot_update_intv", "n_intervals"),
    State("store__saw_position", "data"))
def updateSawLive(n_intervals, pos_data):
    curr_time = current_time() - PLOT_DELAY
    times = [seq['time'] for seq in pos_data]
    for i, time in enumerate(times):
        if time > curr_time:
            curr_seq = i-1 if i > 0 else -1
            return getCoordinateParams(pos_data[curr_seq]), removePlottedSequences(pos_data, curr_seq)
        
    if len(pos_data) > 0:
        return getCoordinateParams(pos_data[-1]), removePlottedSequences(pos_data, -1)
    
    raise PreventUpdate #no new data to plot

def getCoordinateParams(sequence: dict):
    """Returns the parameters for `extendData` with data points taken from the given sequence."""
    markers = list(sequence['markers'].values())
    new_x, new_y, new_z = zip(*markers)
    log.debug(f"Updating saw to sequence {sequence['id']}, first marker coords are: ({new_x[0]}, {new_y[0]}, {new_z[0]})")
    return dict(x=[new_x], y=[new_y], z=[new_z]), [0], len(markers)

def removePlottedSequences(data: list, keep_seq_index: int):
    """Returns a `Patch` with all sequences removed up to the `keep_seq_index`."""
    patch = Patch()
    # if len(data) <= 1: #Check if data is already minimum size
    #     return patch
    # for seq in data[:keep_seq_index]:
    #     patch.remove(seq)
    return patch

@app.callback(
    Output("store__saw_position", "data", allow_duplicate=True),
    Input("data_request_intv", "n_intervals"),
    State("store__saw_position", "data"),
    prevent_initial_call=True)
def getSawPositionData(n_intervals, data: list):
    patch = Patch()
    if len(data) > 0:
        last_seq = max([s['id'] for s in data])
    if (not len(data) > 0) or last_seq == 0:
        last_seq = db.getMaxPrimaryKey(MOCAPSEQ_TBL_NAME)-1
    new_sequences = getSawPositions(last_seq)
    log.debug(f"Added {len(new_sequences)} sequences to the data store (length = {len(data)}).")
    for new_seq in new_sequences:
        patch.append(new_seq)
    return patch

def getSawPositions(last_seq)-> list:
    """Gets the latest saw positions from the database."""
    sequences = getSequencesFromDB(last_seq)
    mocap_data = db.sql(f"SELECT * FROM {MOCAP_TBL_NAME} WHERE `sequence` > %(val)s", dict(val=last_seq))
    log.debug(f"Obtained information for {len(mocap_data)} markers from DB.")
    getSeqByKey = lambda key : [s for s in sequences if s['id'] == key][0]

    for row in mocap_data:
        setMarkerInSequence(*row[1:], getSeqByKey)
    if len(sequences) > 0:
        log.debug(f"Processed {len(sequences)} sequences from DB, first sequence is {sequences[0]}")
    return sequences

def getSequencesFromDB(last_seq)-> list:
    """Returns sequences in the database with a key greater than `last_seq`"""
    db_seqs = db.sql(f"Select * FROM `{MOCAPSEQ_TBL_NAME}` WHERE `sequence_pk` > %(val)s", dict(val=last_seq))
    log.debug(f"Obtained {len(db_seqs)} sequences from DB, last sequence was {last_seq}")
    sequences = list()
    for s in db_seqs:
        seq = dict(id=s[0], markers=dict(), time=time2Float(s[1]))
        sequences.append(seq)
    return sequences

def setMarkerInSequence(label: str, val: float, seq_key: int, 
                        marker_key: int, getSeqByKey)-> dict:
    """Adds the marker to the correct Sequence dictionary in a list of sequences"""
    seq = getSeqByKey(seq_key)
    if marker_key not in seq['markers']: 
        seq['markers'][marker_key] = [0]*3
    label_idx = ['x', 'y', 'z'].index(label)
    seq['markers'][marker_key][label_idx] = val

def time2Float(time_in: str)-> float:
    """Converts the inputted time to Epoch time."""
    if isinstance(time_in, float): return time_in
    epoch = float(time_in.strftime('%s.%f'))
    return epoch
        
def run():
    app.run_server(debug=False, dev_tools_silence_routes_logging=True)

if __name__ == '__main__':
    run()

"""
Sequence Dictionary Description
Container class for set of coordinates taken at a single timestamp.
    Parameters
    ----------
    id : int
        The primary key for the sequence.
    markers : dict
        A dict mapping a primary key (int) to a 1x3 list of coordinates, as 
        in `markers = {1: [0.1, 1.1, 2.2], 3: [1.2, 0.1, 3.2]}`.
    time : float
        The epoch time the sequence was observed.
"""

