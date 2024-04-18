from dash import Dash, dcc, html, Input, Output, State, Patch
from dash.exceptions import PreventUpdate
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from time import mktime, strptime, time as current_time

from src.db.reader import Reader
from src.db.db_names import *

PLOT_DELAY = 3 #seconds
app = Dash(update_title=None)
db = Reader()

fig__saw_live = go.Figure(data=[go.Scatter3d(
    x=[0, 0, 1], y=[0, 0, 1], z=[0, 1, 1], 
    mode='lines+markers',
    line= dict(width = 20))])
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
    dcc.Store(id='store__saw_position', data=[
        dict( #Dummy data
            sequence=0, 
            markers={
                1: [0,0,0],
                2: [1,1,1]}, 
            time=0.0) ]),
    # dcc.Loading(
        dcc.Graph(id="graph", figure=fig__saw_live), 
        # type='dot', color='#F56600'),
    dcc.Interval(id='data_request_intv', interval=2000),
    dcc.Interval(id='plot_update_intv', interval=1000)
])

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
            curr_seq = i-1 if i > 0 else 0
            return getCoordinateParams(pos_data[curr_seq]), removePlottedSequences(pos_data, curr_seq)
        
    if len(pos_data) > 0:
        return getCoordinateParams(pos_data[-1]), removePlottedSequences(pos_data, -1)
    
    raise PreventUpdate #no new data to plot

def getCoordinateParams(sequence_data: dict):
    """Returns a the parameters for `extendData` with data points taken from the given sequence."""
    patch = Patch()
    markers = list(sequence_data['markers'].values())
    new_x, new_y, new_z = zip(*markers)
    return dict(x=[new_x], y=[new_y], z=[new_z]), [0], len(markers)

def removePlottedSequences(data: list, keep_seq_index: int):
    """Returns a `Patch` with all sequences removed up to the `keep_seq_index`."""
    patch = Patch()
    for seq in data[:keep_seq_index]:
        patch.remove(seq)
    return patch

@app.callback(
    Output("store__saw_position", "data", allow_duplicate=True),
    Input("data_request_intv", "n_intervals"),
    State("store__saw_position", "data"),
    prevent_initial_call=True)
def getSawPositionData(n_intervals, data):
    patch = Patch()
    last_seq = max([s['sequence'] for s in data])
    new_sequences = getSawPositions(last_seq)
    patch.extend(new_sequences)
    return patch

def getSawPositions(last_seq)-> list:
    """Gets the latest saw positions from the database."""
    addMarkers(np.random.randint(0, 3))
    seqs = db.sql(f"Select * FROM `MocapSequences` WHERE `sequence_pk` > %(val)s", dict(val=last_seq))
    mocap_data = db.sql(f"SELECT * FROM `Mocap` WHERE `sequence` > %(val)s", dict(val=last_seq))
    sequences = [dict(sequence=s[0], markers=dict(), time=time2Float(s[1])) for s in seqs]
    get_sequence = lambda seq_key : [s for s in sequences if s['sequence'] == seq_key][0]
    for row in mocap_data:
        label, val, seq_pk, marker = row[1:]
        sequence = sequences[sequences.index(get_sequence(seq_pk))]
        markers = sequence['markers']
        if marker not in markers: markers[marker] = [0]*3
        markers[marker][['x', 'y', 'z'].index(label)] = val
    return sequences

def time2Float(time_in: str)-> float:
    """Converts the inputted time to Epoch time."""
    if isinstance(time_in, float): return time_in
    epoch = float(time_in.strftime('%s.%f'))
    return epoch

##### DEV #######
from src.db.db_ops import addEntry, getEntries

def addMarkers(n: int=1):
    entries = getEntries(db.csr, "MocapSequences", "sequence_pk")
    last_seq = max([i[0] for i in entries])
    for i in range(n):
        seq = last_seq + i + 1
        addEntry(db.csr, "MocapSequences", [seq], ["sequence_pk"])
        entries = getEntries(db.csr, "Mocap", "mocap_pk")
        pk = max([i[0] for i in entries]) + 1

        for mrkr in [1, 2]:
            for label in ['x', 'y', 'z']:
                value = np.random.rand() * 6
                addEntry(db.csr, "Mocap", 
                         [pk, label, value, seq, mrkr], 
                         ['mocap_pk', 'label', 'value', 'sequence', 'marker_pk'])
                pk += 1
#################
        
def run():
    app.run_server(debug=False)

if __name__ == '__main__':
    run()

