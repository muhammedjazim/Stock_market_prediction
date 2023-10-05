#necessary libraries
import yfinance as yf
import datetime as dt
import pandas as pd
import plotly.graph_objs as go
import pandas_datareader.data as pdr
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import tensorflow as tf
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from tqdm import tqdm
tf.compat.v1.disable_eager_execution()
from flask import Flask, render_template, request

app = Flask(__name__)

yf.pdr_override() #yfinance data over pandas

@app.route('/', methods=['GET', 'POST'])
def index():
        if request.method == 'POST':
            stock_symbol = request.form['stock_symbol'].upper()
            stock_name = f"{stock_symbol}.BO"
            end = dt.datetime.now()
            start = end - dt.timedelta(days=365)
                
            try:    
                df = pdr.DataReader(stock_name, start, end)
                df.reset_index(inplace=True)
            except Exception as e:
                error_message = f"Error: {e}. Please enter a valid stock name."
                return render_template('index.html', error=error_message)
        
            if "Symbol" in df.columns:
                df = df.drop("Symbol", axis=1)
            
            #Pre processing
            minmax = MinMaxScaler().fit(df[['Close']].astype('float32'))
            df_log = minmax.transform(df[['Close']].astype('float32'))
            df_log = pd.DataFrame(df_log)

            #defining hyper parameters
            simulation_size = 10
            num_layers = 1
            size_layer = 128
            timestamp = 5
            epoch = 300
            dropout_rate = 0.8
            test_size = 30
            learning_rate = 0.01

            df_train = df_log

            #define model with my neural network architecture
            class Model:
                def __init__(
                    self,
                    learning_rate,
                    num_layers,
                    size,
                    size_layer,
                    output_size,
                    forget_bias=0.1,
                ):
                    def lstm_cell(size_layer):
                        return tf.compat.v1.nn.rnn_cell.LSTMCell(size_layer, state_is_tuple=False)

                    rnn_cells = tf.compat.v1.nn.rnn_cell.MultiRNNCell(
                        [lstm_cell(size_layer) for _ in range(num_layers)],
                        state_is_tuple=False,
                    )
                    self.X = tf.compat.v1.placeholder(tf.float32, (None, None, size))
                    self.Y = tf.compat.v1.placeholder(tf.float32, (None, output_size))
                    drop = tf.compat.v1.nn.rnn_cell.DropoutWrapper(
                        rnn_cells, output_keep_prob=forget_bias
                    )
                    self.hidden_layer = tf.compat.v1.placeholder(
                        tf.float32, (None, num_layers * 2 * size_layer)
                    )
                    self.outputs, self.last_state = tf.compat.v1.nn.dynamic_rnn(
                        drop, self.X, initial_state=self.hidden_layer, dtype=tf.float32
                    )
                    self.logits = tf.compat.v1.layers.dense(self.outputs[:, -1], output_size)
                    self.cost = tf.reduce_mean(tf.square(self.Y - self.logits))
                    self.optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate).minimize(
                        self.cost
                    )

            #calculate accuracy
            def calculate_accuracy(real, predict):
                real = np.array(real) + 1
                predict = np.array(predict) + 1
                percentage = 1 - np.sqrt(np.mean(np.square((real - predict) / real)))
                return percentage * 100

            #smooth sequence of data points using weighted moving average
            #using weights and data points,claculate weihted average
            #like moving averages, reducing short term fluctuations and taking long term trends
            def anchor(signal, weight):
                buffer = []
                last = signal[0]
                for i in signal:
                    smoothed_val = last * weight + (1 - weight) * i
                    buffer.append(smoothed_val)
                    last = smoothed_val
                return buffer

            #training and keep track of loss and accuracy
            def forecast(
                learning_rate, num_layers, size_layer, dropout_rate, epoch, timestamp, test_size
            ):
                tf.compat.v1.reset_default_graph()
                modelnn = Model(
                    learning_rate,
                    num_layers,
                    df_log.shape[1],
                    size_layer,
                    df_log.shape[1],
                    dropout_rate,
                )
                sess = tf.compat.v1.InteractiveSession()
                sess.run(tf.compat.v1.global_variables_initializer())
                date_ori = pd.to_datetime(df.iloc[:, 0]).tolist()

                pbar = tqdm(range(epoch), desc="train loop")
                for i in pbar:
                    init_value = np.zeros((1, num_layers * 2 * size_layer))
                    total_loss, total_acc = [], []
                    for k in range(0, df_train.shape[0] - 1, timestamp):
                        index = min(k + timestamp, df_train.shape[0] - 1)
                        batch_x = np.expand_dims(df_train.iloc[k: index, :].values, axis=0)
                        batch_y = df_train.iloc[k + 1: index + 1, :].values
                        logits, last_state, _, loss = sess.run(
                            [
                                modelnn.logits,
                                modelnn.last_state,
                                modelnn.optimizer,
                                modelnn.cost,
                            ],
                            feed_dict={
                                modelnn.X: batch_x,
                                modelnn.Y: batch_y,
                                modelnn.hidden_layer: init_value,
                            },
                        )

                        init_value = last_state
                        total_loss.append(loss)
                        total_acc.append(calculate_accuracy(batch_y[:, 0], logits[:, 0]))
                    pbar.set_postfix(cost=np.mean(total_loss), acc=np.mean(total_acc))

                #forecasting future prices using our trained model
                future_day = test_size

                output_predict = np.zeros((df_train.shape[0] + future_day, df_train.shape[1]))
                output_predict[0] = df_train.iloc[0]
                upper_b = (df_train.shape[0] // timestamp) * timestamp
                init_value = np.zeros((1, num_layers * 2 * size_layer))

                for k in range(0, (df_train.shape[0] // timestamp) * timestamp, timestamp):
                    out_logits, last_state = sess.run(
                        [modelnn.logits, modelnn.last_state],
                        feed_dict={
                            modelnn.X: np.expand_dims(
                                df_train.iloc[k: k + timestamp], axis=0
                            ),
                            modelnn.hidden_layer: init_value,
                        },
                    )
                    init_value = last_state
                    output_predict[k + 1: k + timestamp + 1] = out_logits

                if upper_b != df_train.shape[0]:
                    out_logits, last_state = sess.run(
                        [modelnn.logits, modelnn.last_state],
                        feed_dict={
                            modelnn.X: np.expand_dims(df_train.iloc[upper_b:], axis=0),
                            modelnn.hidden_layer: init_value,
                        },
                    )
                    output_predict[upper_b + 1: df_train.shape[0] + 1] = out_logits
                    future_day -= 1
                    date_ori.append(date_ori[-1] + dt.timedelta(days=1))

                init_value = last_state

                for i in range(future_day):
                    o = output_predict[-future_day - timestamp + i: -future_day + i]
                    out_logits, last_state = sess.run(
                        [modelnn.logits, modelnn.last_state],
                        feed_dict={
                            modelnn.X: np.expand_dims(o, axis=0),
                            modelnn.hidden_layer: init_value,
                        },
                    )
                    init_value = last_state
                    output_predict[-future_day + i] = out_logits[-1]
                    date_ori.append(date_ori[-1] + dt.timedelta(days=1))

                output_predict = minmax.inverse_transform(output_predict)
                deep_future = anchor(output_predict[:, 0], 0.4)

                return deep_future

            #run multiple simulations of predictions and store each values
            results = []

            for i in range(simulation_size):
                print('Simulation %d' % (i + 1))
                forecast_result = forecast(
                    learning_rate=learning_rate,
                    num_layers=num_layers,
                    size_layer=size_layer,
                    dropout_rate=dropout_rate,
                    epoch=epoch,
                    timestamp=timestamp,
                    test_size=test_size
                )
                results.append(forecast_result)

            date_ori = pd.to_datetime(df.iloc[:, 0]).tolist()
            for i in range(test_size):
                date_ori.append(date_ori[-1] + dt.timedelta(days=1))
            date_ori = pd.Series(date_ori).dt.strftime(date_format='%Y-%m-%d').tolist()

            #remove unrealistic predictions
            #check if predicted values go both beyond lower limit and upper limit of originals a lot
            #then reject that trend
            accepted_results = []
            for r in results:
                if (np.array(r[-test_size:]) < np.min(df['Close'])).sum() == 0 and \
                (np.array(r[-test_size:]) > np.max(df['Close']) * 2).sum() == 0:
                    accepted_results.append(r)

            #let's store forecasted values of each simulation
            forecast_arrays = []

            for forecast_result in results:
                forecast_arrays.append(forecast_result[-test_size:])

            #visualize original and predicted
            import plotly.express as px
            import plotly.graph_objects as go

            accuracies = [calculate_accuracy(df['Close'].values, r[:-test_size]) for r in accepted_results]

            fig = go.Figure()

            fig.add_trace(go.Scatter(
                x=np.arange(len(max(accepted_results))),
                y=max(accepted_results),
                name='Prediction',
                hovertemplate='Date: %{text}<br>Price: ₹%{y:.2f}',
                text=[date_ori[i] for i in range(len(max(accepted_results)))]
            ))

            fig.add_trace(go.Scatter(
                x=np.arange(len(df['Close'])),
                y=df['Close'],
                name='true trend',
                mode='lines',
                line_color='black',
                hovertemplate='Date: %{text}<br>Price: ₹%{y:.2f}',
                text=date_ori
            ))

            fig.update_layout(
                legend=dict(title='Legend'),
                title='Prediction'
            )
            x_range_future = np.arange(len(results[0]))
            fig.update_layout(
                xaxis=dict(
                    tickvals=x_range_future[::30],
                    ticktext=date_ori[::30],
                )
            )

            picture1 = fig.to_json()

        
            return render_template('result.html', fig=picture1)

        return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
