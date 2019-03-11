import dash
import dash_core_components as dcc
import dash_html_components as html
import pandas as pd
import plotly.graph_objs as go

app = dash.Dash(__name__)

_path = 'C:\\Users\\ccamp\Documents\\GitHub\\HerbariumDeep\\frameworks\\TensorFlow\\TFHub\\tmp\\summaries\\hyperparams.pkl'
df = pd.read_pickle(_path)
grid_search_text_list = list(
    zip(
        'init: ' + df.initializer.apply(str),
        'optim: ' + df.optimizer.apply(str),
        'activ: ' + df.activation.apply(str),
        'fit time (sec): ' + df.fit_time_sec.apply(str),
        'mean acc: ' + df.mean_acc.apply(str),
        'mean loss: ' + df.mean_loss.apply(str),
        'mean top-k (k=5) acc: ' + df.mean_top_five_acc.apply(str)
    )
)
text = ['<br>'.join(t) for t in grid_search_text_list]
df['Text'] = text

app.layout = html.Div(
    children=[html.H1(children='Hello Dash'),
    html.Div(children='''Dash: A web appplication framework for Python.'''),
    # dcc.Graph(
    #     id='example-graph',
    #     figure={
    #             'data': [
    #                 {'x': [1, 2, 3], 'y': [4, 1, 2], 'type': 'bar', 'name': 'SF'},
    #                 {'x': [1, 2, 3], 'y': [2, 4, 5], 'type': 'bar', 'name': u'Montréal'},
    #             ],
    #             'layout': {
    #                 'title': 'Dash Data Visualization'
    #             }
    #     }
    # )

    dcc.Graph(
        id='my-graph',
        figure={
            'data': [
                go.Scatter3d(
                    x=df['initializer'],
                    y=df['optimizer'],
                    z=df['activation'],
                    mode='markers',
                    marker=dict(
                        # size=mean_fit_time ** (1 / 3),
                        color=df.mean_acc,
                        opacity=0.99,
                        colorscale='Viridis',
                        colorbar=dict(title='Mean Acc', tickmode='auto', nticks=10),
                        line=dict(color='rgb(140, 140, 170)')
                    ),
                    text=df.Text,
                    hoverinfo='text'
                )
            ],
            'layout': [
                go.Layout(
                    title='3D Scatter Plot of Grid Search Results',
                    margin=dict(
                        l=30,
                        r=30,
                        b=30,
                        t=30
                    ),
                    # height=600,
                    # width=960,
                    scene=dict(
                        xaxis=dict(
                            title='initializer',
                            nticks=10
                        ),
                        yaxis=dict(
                            title='optimizer'
                        ),
                        zaxis=dict(
                            title='activation'
                        ),
                    )
                )
            ]
        }
    )
])



if __name__ == '__main__':
    app.run_server(debug=True)
