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
num_unique_optimizers = len(df['optimizer'].unique())
optims_one_hot = pd.get_dummies(df.optimizer)

app.layout = html.Div(children=[
    html.H1(children='Parallel Coordinates Plot'),
    html.Div(children=[
              dcc.Graph(
                  id='my-graph',
                  style={'height': '100%'},
                  figure={
                      'data': [
                          go.Parcoords(
                              line=dict(color=df.mean_acc),
                              dimensions=list([
                                  dict(
                                      range=[0, 30], constraintrange=[0, 10], label='train_batch_size', values=df.train_batch_size
                                  ),
                                  # dict(
                                  #     range=[0, num_unique_optimizers], constraintrange=[0, 1], label='optimizer', tickvals=optims_one_hot.OPTIM_ADAM.values, ticktext=['ADAM' if one_hot == 1 else 'NESTEROV' for one_hot in optims_one_hot.OPTIM_ADAM.values]
                                  # )
                                  dict(
                                      range=[0, 2], label='optimizer', tickvals=[0, 1], ticktext=['adam', 'nesterov']
                                  )
                              ])
                          )
                      ],
                      'layout': [
                          go.Layout(
                              plot_bgcolor='#E5E5E5',
                              paper_bgcolor='White'
                          )
                      ]
                  }
              )
        ])
    ]
)

if __name__ == '__main__':
    app.run_server(debug=True)
