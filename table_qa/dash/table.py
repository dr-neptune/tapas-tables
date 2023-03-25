from dash import Dash, dcc, html, Input, Output, dash_table

# Initialize the app
app = Dash(__name__)

# Define the app layout
app.layout = html.Div([
    dash_table.DataTable(
        id='data-table',
        columns=[
            {'name': 'Column A', 'id': 'column-a'},
            {'name': 'Column B', 'id': 'column-b'}
        ],
        data=[
            {'column-a': 'Row 1 A', 'column-b': 'Row 1 B'},
            {'column-a': 'Row 2 A', 'column-b': 'Row 2 B'}
        ],
        style_cell={'textAlign': 'left'},
        style_data={'whiteSpace': 'normal'},
        css=[{
            'selector': '.dash-cell div.dash-cell-value',
            'rule': 'display: inline; white-space: inherit; overflow: inherit; text-overflow: inherit;'
        }]
    ),
    html.Br(),
    html.Div(id='text-output'),
    html.Br(),
    dcc.Input(id='text-input', type='text', placeholder='Enter some text...'),
])

# Define the callback to update the text-output Div
@app.callback(
    Output('text-output', 'children'),
    Input('text-input', 'value')
)
def update_output(value):
    if value is None or value == '':
        return 'Text input is empty.'
    else:
        return f'You entered: {value}'

if __name__ == '__main__':
    app.run_server(debug=True)
