{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 1,
   "metadata": {},
   "outputs": [],
   "source": [
    "import sys\n",
    "import warnings\n",
    "\n",
    "if not sys.warnoptions:\n",
    "    warnings.simplefilter('ignore')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 2,
   "metadata": {},
   "outputs": [],
   "source": [
    "import tensorflow as tf\n",
    "import numpy as np\n",
    "import matplotlib.pyplot as plt\n",
    "import seaborn as sns\n",
    "import pandas as pd\n",
    "from sklearn.preprocessing import MinMaxScaler\n",
    "from datetime import datetime\n",
    "from datetime import timedelta\n",
    "from tqdm import tqdm\n",
    "sns.set()\n",
    "tf.compat.v1.random.set_random_seed(1234)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>Date</th>\n",
       "      <th>Open</th>\n",
       "      <th>High</th>\n",
       "      <th>Low</th>\n",
       "      <th>Close</th>\n",
       "      <th>Adj Close</th>\n",
       "      <th>Volume</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>2016-11-02</td>\n",
       "      <td>778.200012</td>\n",
       "      <td>781.650024</td>\n",
       "      <td>763.450012</td>\n",
       "      <td>768.700012</td>\n",
       "      <td>768.700012</td>\n",
       "      <td>1872400</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>2016-11-03</td>\n",
       "      <td>767.250000</td>\n",
       "      <td>769.950012</td>\n",
       "      <td>759.030029</td>\n",
       "      <td>762.130005</td>\n",
       "      <td>762.130005</td>\n",
       "      <td>1943200</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>2016-11-04</td>\n",
       "      <td>750.659973</td>\n",
       "      <td>770.359985</td>\n",
       "      <td>750.560974</td>\n",
       "      <td>762.020020</td>\n",
       "      <td>762.020020</td>\n",
       "      <td>2134800</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>2016-11-07</td>\n",
       "      <td>774.500000</td>\n",
       "      <td>785.190002</td>\n",
       "      <td>772.549988</td>\n",
       "      <td>782.520020</td>\n",
       "      <td>782.520020</td>\n",
       "      <td>1585100</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>2016-11-08</td>\n",
       "      <td>783.400024</td>\n",
       "      <td>795.632996</td>\n",
       "      <td>780.190002</td>\n",
       "      <td>790.510010</td>\n",
       "      <td>790.510010</td>\n",
       "      <td>1350800</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "         Date        Open        High         Low       Close   Adj Close  \\\n",
       "0  2016-11-02  778.200012  781.650024  763.450012  768.700012  768.700012   \n",
       "1  2016-11-03  767.250000  769.950012  759.030029  762.130005  762.130005   \n",
       "2  2016-11-04  750.659973  770.359985  750.560974  762.020020  762.020020   \n",
       "3  2016-11-07  774.500000  785.190002  772.549988  782.520020  782.520020   \n",
       "4  2016-11-08  783.400024  795.632996  780.190002  790.510010  790.510010   \n",
       "\n",
       "    Volume  \n",
       "0  1872400  \n",
       "1  1943200  \n",
       "2  2134800  \n",
       "3  1585100  \n",
       "4  1350800  "
      ]
     },
     "execution_count": 3,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df = pd.read_csv('../dataset/GOOG-year.csv')\n",
    "df.head()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>0</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>0.112708</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>0.090008</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>0.089628</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>0.160459</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>0.188066</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "          0\n",
       "0  0.112708\n",
       "1  0.090008\n",
       "2  0.089628\n",
       "3  0.160459\n",
       "4  0.188066"
      ]
     },
     "execution_count": 4,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "minmax = MinMaxScaler().fit(df.iloc[:, 4:5].astype('float32')) # Close index\n",
    "df_log = minmax.transform(df.iloc[:, 4:5].astype('float32')) # Close index\n",
    "df_log = pd.DataFrame(df_log)\n",
    "df_log.head()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Forecast\n",
    "\n",
    "This example is using model 1.lstm, if you want to use another model, need to tweak a little bit, but I believe it is not that hard.\n",
    "\n",
    "I want to forecast 30 days ahead! So just change `test_size` to forecast `t + N` ahead.\n",
    "\n",
    "Also, I want to simulate 10 times, 10 variances of forecasted patterns. Just change `simulation_size`."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 5,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "((252, 7), (252, 1))"
      ]
     },
     "execution_count": 5,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "simulation_size = 10\n",
    "num_layers = 1\n",
    "size_layer = 128\n",
    "timestamp = 5\n",
    "epoch = 300\n",
    "dropout_rate = 0.8\n",
    "test_size = 30\n",
    "learning_rate = 0.01\n",
    "\n",
    "df_train = df_log\n",
    "df.shape, df_train.shape"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 6,
   "metadata": {},
   "outputs": [],
   "source": [
    "class Model:\n",
    "    def __init__(\n",
    "        self,\n",
    "        learning_rate,\n",
    "        num_layers,\n",
    "        size,\n",
    "        size_layer,\n",
    "        output_size,\n",
    "        forget_bias = 0.1,\n",
    "    ):\n",
    "        def lstm_cell(size_layer):\n",
    "            return tf.nn.rnn_cell.LSTMCell(size_layer, state_is_tuple = False)\n",
    "\n",
    "        rnn_cells = tf.nn.rnn_cell.MultiRNNCell(\n",
    "            [lstm_cell(size_layer) for _ in range(num_layers)],\n",
    "            state_is_tuple = False,\n",
    "        )\n",
    "        self.X = tf.placeholder(tf.float32, (None, None, size))\n",
    "        self.Y = tf.placeholder(tf.float32, (None, output_size))\n",
    "        drop = tf.contrib.rnn.DropoutWrapper(\n",
    "            rnn_cells, output_keep_prob = forget_bias\n",
    "        )\n",
    "        self.hidden_layer = tf.placeholder(\n",
    "            tf.float32, (None, num_layers * 2 * size_layer)\n",
    "        )\n",
    "        self.outputs, self.last_state = tf.nn.dynamic_rnn(\n",
    "            drop, self.X, initial_state = self.hidden_layer, dtype = tf.float32\n",
    "        )\n",
    "        self.logits = tf.layers.dense(self.outputs[-1], output_size)\n",
    "        self.cost = tf.reduce_mean(tf.square(self.Y - self.logits))\n",
    "        self.optimizer = tf.train.AdamOptimizer(learning_rate).minimize(\n",
    "            self.cost\n",
    "        )\n",
    "        \n",
    "def calculate_accuracy(real, predict):\n",
    "    real = np.array(real) + 1\n",
    "    predict = np.array(predict) + 1\n",
    "    percentage = 1 - np.sqrt(np.mean(np.square((real - predict) / real)))\n",
    "    return percentage * 100\n",
    "\n",
    "def anchor(signal, weight):\n",
    "    buffer = []\n",
    "    last = signal[0]\n",
    "    for i in signal:\n",
    "        smoothed_val = last * weight + (1 - weight) * i\n",
    "        buffer.append(smoothed_val)\n",
    "        last = smoothed_val\n",
    "    return buffer"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 7,
   "metadata": {},
   "outputs": [],
   "source": [
    "def forecast():\n",
    "    tf.reset_default_graph()\n",
    "    modelnn = Model(\n",
    "        learning_rate, num_layers, df_log.shape[1], size_layer, df_log.shape[1], dropout_rate\n",
    "    )\n",
    "    sess = tf.InteractiveSession()\n",
    "    sess.run(tf.global_variables_initializer())\n",
    "    date_ori = pd.to_datetime(df.iloc[:, 0]).tolist()\n",
    "\n",
    "    pbar = tqdm(range(epoch), desc = 'train loop')\n",
    "    for i in pbar:\n",
    "        init_value = np.zeros((1, num_layers * 2 * size_layer))\n",
    "        total_loss, total_acc = [], []\n",
    "        for k in range(0, df_train.shape[0] - 1, timestamp):\n",
    "            index = min(k + timestamp, df_train.shape[0] - 1)\n",
    "            batch_x = np.expand_dims(\n",
    "                df_train.iloc[k : index, :].values, axis = 0\n",
    "            )\n",
    "            batch_y = df_train.iloc[k + 1 : index + 1, :].values\n",
    "            logits, last_state, _, loss = sess.run(\n",
    "                [modelnn.logits, modelnn.last_state, modelnn.optimizer, modelnn.cost],\n",
    "                feed_dict = {\n",
    "                    modelnn.X: batch_x,\n",
    "                    modelnn.Y: batch_y,\n",
    "                    modelnn.hidden_layer: init_value,\n",
    "                },\n",
    "            )        \n",
    "            init_value = last_state\n",
    "            total_loss.append(loss)\n",
    "            total_acc.append(calculate_accuracy(batch_y[:, 0], logits[:, 0]))\n",
    "        pbar.set_postfix(cost = np.mean(total_loss), acc = np.mean(total_acc))\n",
    "    \n",
    "    future_day = test_size\n",
    "\n",
    "    output_predict = np.zeros((df_train.shape[0] + future_day, df_train.shape[1]))\n",
    "    output_predict[0] = df_train.iloc[0]\n",
    "    upper_b = (df_train.shape[0] // timestamp) * timestamp\n",
    "    init_value = np.zeros((1, num_layers * 2 * size_layer))\n",
    "\n",
    "    for k in range(0, (df_train.shape[0] // timestamp) * timestamp, timestamp):\n",
    "        out_logits, last_state = sess.run(\n",
    "            [modelnn.logits, modelnn.last_state],\n",
    "            feed_dict = {\n",
    "                modelnn.X: np.expand_dims(\n",
    "                    df_train.iloc[k : k + timestamp], axis = 0\n",
    "                ),\n",
    "                modelnn.hidden_layer: init_value,\n",
    "            },\n",
    "        )\n",
    "        init_value = last_state\n",
    "        output_predict[k + 1 : k + timestamp + 1] = out_logits\n",
    "\n",
    "    if upper_b != df_train.shape[0]:\n",
    "        out_logits, last_state = sess.run(\n",
    "            [modelnn.logits, modelnn.last_state],\n",
    "            feed_dict = {\n",
    "                modelnn.X: np.expand_dims(df_train.iloc[upper_b:], axis = 0),\n",
    "                modelnn.hidden_layer: init_value,\n",
    "            },\n",
    "        )\n",
    "        output_predict[upper_b + 1 : df_train.shape[0] + 1] = out_logits\n",
    "        future_day -= 1\n",
    "        date_ori.append(date_ori[-1] + timedelta(days = 1))\n",
    "\n",
    "    init_value = last_state\n",
    "    \n",
    "    for i in range(future_day):\n",
    "        o = output_predict[-future_day - timestamp + i:-future_day + i]\n",
    "        out_logits, last_state = sess.run(\n",
    "            [modelnn.logits, modelnn.last_state],\n",
    "            feed_dict = {\n",
    "                modelnn.X: np.expand_dims(o, axis = 0),\n",
    "                modelnn.hidden_layer: init_value,\n",
    "            },\n",
    "        )\n",
    "        init_value = last_state\n",
    "        output_predict[-future_day + i] = out_logits[-1]\n",
    "        date_ori.append(date_ori[-1] + timedelta(days = 1))\n",
    "    \n",
    "    output_predict = minmax.inverse_transform(output_predict)\n",
    "    deep_future = anchor(output_predict[:, 0], 0.4)\n",
    "    \n",
    "    return deep_future"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 8,
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "WARNING: Logging before flag parsing goes to stderr.\n",
      "W0818 12:00:52.795618 140214804277056 deprecation.py:323] From <ipython-input-6-d01d21f09afe>:12: LSTMCell.__init__ (from tensorflow.python.ops.rnn_cell_impl) is deprecated and will be removed in a future version.\n",
      "Instructions for updating:\n",
      "This class is equivalent as tf.keras.layers.LSTMCell, and will be replaced by that in Tensorflow 2.0.\n",
      "W0818 12:00:52.799092 140214804277056 rnn_cell_impl.py:893] <tensorflow.python.ops.rnn_cell_impl.LSTMCell object at 0x7f8644897400>: Using a concatenated state is slower and will soon be deprecated.  Use state_is_tuple=True.\n",
      "W0818 12:00:52.801252 140214804277056 deprecation.py:323] From <ipython-input-6-d01d21f09afe>:16: MultiRNNCell.__init__ (from tensorflow.python.ops.rnn_cell_impl) is deprecated and will be removed in a future version.\n",
      "Instructions for updating:\n",
      "This class is equivalent as tf.keras.layers.StackedRNNCells, and will be replaced by that in Tensorflow 2.0.\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "simulation 1\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "W0818 12:00:53.121960 140214804277056 lazy_loader.py:50] \n",
      "The TensorFlow contrib module will not be included in TensorFlow 2.0.\n",
      "For more information, please see:\n",
      "  * https://github.com/tensorflow/community/blob/master/rfcs/20180907-contrib-sunset.md\n",
      "  * https://github.com/tensorflow/addons\n",
      "  * https://github.com/tensorflow/io (for I/O related ops)\n",
      "If you depend on functionality not listed there, please file an issue.\n",
      "\n",
      "W0818 12:00:53.125179 140214804277056 deprecation.py:323] From <ipython-input-6-d01d21f09afe>:27: dynamic_rnn (from tensorflow.python.ops.rnn) is deprecated and will be removed in a future version.\n",
      "Instructions for updating:\n",
      "Please use `keras.layers.RNN(cell)`, which is equivalent to this API\n",
      "W0818 12:00:53.314420 140214804277056 deprecation.py:506] From /usr/local/lib/python3.6/dist-packages/tensorflow/python/ops/init_ops.py:1251: calling VarianceScaling.__init__ (from tensorflow.python.ops.init_ops) with dtype is deprecated and will be removed in a future version.\n",
      "Instructions for updating:\n",
      "Call initializer instance with the dtype argument instead of passing it to the constructor\n",
      "W0818 12:00:53.321002 140214804277056 deprecation.py:506] From /usr/local/lib/python3.6/dist-packages/tensorflow/python/ops/rnn_cell_impl.py:961: calling Zeros.__init__ (from tensorflow.python.ops.init_ops) with dtype is deprecated and will be removed in a future version.\n",
      "Instructions for updating:\n",
      "Call initializer instance with the dtype argument instead of passing it to the constructor\n",
      "W0818 12:00:53.718872 140214804277056 deprecation.py:323] From <ipython-input-6-d01d21f09afe>:29: dense (from tensorflow.python.layers.core) is deprecated and will be removed in a future version.\n",
      "Instructions for updating:\n",
      "Use keras.layers.dense instead.\n",
      "train loop: 100%|██████████| 300/300 [01:17<00:00,  3.90it/s, acc=95.9, cost=0.00437]\n",
      "W0818 12:02:12.766668 140214804277056 rnn_cell_impl.py:893] <tensorflow.python.ops.rnn_cell_impl.LSTMCell object at 0x7f85be966eb8>: Using a concatenated state is slower and will soon be deprecated.  Use state_is_tuple=True.\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "simulation 2\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "train loop: 100%|██████████| 300/300 [01:18<00:00,  3.81it/s, acc=96.2, cost=0.00386]\n",
      "W0818 12:03:31.524121 140214804277056 rnn_cell_impl.py:893] <tensorflow.python.ops.rnn_cell_impl.LSTMCell object at 0x7f85b4c59dd8>: Using a concatenated state is slower and will soon be deprecated.  Use state_is_tuple=True.\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "simulation 3\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "train loop: 100%|██████████| 300/300 [01:17<00:00,  3.86it/s, acc=95.9, cost=0.00421]\n",
      "W0818 12:04:49.292782 140214804277056 rnn_cell_impl.py:893] <tensorflow.python.ops.rnn_cell_impl.LSTMCell object at 0x7f85ac67f5f8>: Using a concatenated state is slower and will soon be deprecated.  Use state_is_tuple=True.\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "simulation 4\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "train loop: 100%|██████████| 300/300 [01:17<00:00,  3.85it/s, acc=95.1, cost=0.00617]\n",
      "W0818 12:06:07.690939 140214804277056 rnn_cell_impl.py:893] <tensorflow.python.ops.rnn_cell_impl.LSTMCell object at 0x7f85209545f8>: Using a concatenated state is slower and will soon be deprecated.  Use state_is_tuple=True.\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "simulation 5\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "train loop: 100%|██████████| 300/300 [01:18<00:00,  3.81it/s, acc=96.8, cost=0.00293]\n",
      "W0818 12:07:26.842436 140214804277056 rnn_cell_impl.py:893] <tensorflow.python.ops.rnn_cell_impl.LSTMCell object at 0x7f85089d1128>: Using a concatenated state is slower and will soon be deprecated.  Use state_is_tuple=True.\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "simulation 6\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "train loop: 100%|██████████| 300/300 [01:17<00:00,  3.82it/s, acc=97.3, cost=0.00178]\n",
      "W0818 12:08:45.222193 140214804277056 rnn_cell_impl.py:893] <tensorflow.python.ops.rnn_cell_impl.LSTMCell object at 0x7f85082c6160>: Using a concatenated state is slower and will soon be deprecated.  Use state_is_tuple=True.\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "simulation 7\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "train loop: 100%|██████████| 300/300 [01:16<00:00,  3.94it/s, acc=97.5, cost=0.00161]\n",
      "W0818 12:10:01.933482 140214804277056 rnn_cell_impl.py:893] <tensorflow.python.ops.rnn_cell_impl.LSTMCell object at 0x7f84fc7de208>: Using a concatenated state is slower and will soon be deprecated.  Use state_is_tuple=True.\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "simulation 8\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "train loop: 100%|██████████| 300/300 [01:17<00:00,  3.81it/s, acc=97.5, cost=0.00156]\n",
      "W0818 12:11:20.348971 140214804277056 rnn_cell_impl.py:893] <tensorflow.python.ops.rnn_cell_impl.LSTMCell object at 0x7f84fc7127b8>: Using a concatenated state is slower and will soon be deprecated.  Use state_is_tuple=True.\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "simulation 9\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "train loop: 100%|██████████| 300/300 [01:18<00:00,  3.81it/s, acc=96.7, cost=0.00297]\n",
      "W0818 12:12:39.812369 140214804277056 rnn_cell_impl.py:893] <tensorflow.python.ops.rnn_cell_impl.LSTMCell object at 0x7f84f6ed44a8>: Using a concatenated state is slower and will soon be deprecated.  Use state_is_tuple=True.\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "simulation 10\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "train loop: 100%|██████████| 300/300 [01:17<00:00,  3.98it/s, acc=97.5, cost=0.00179]\n"
     ]
    }
   ],
   "source": [
    "results = []\n",
    "for i in range(simulation_size):\n",
    "    print('simulation %d'%(i + 1))\n",
    "    results.append(forecast())"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 10,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "['2017-11-27', '2017-11-28', '2017-11-29', '2017-11-30', '2017-12-01']"
      ]
     },
     "execution_count": 10,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "date_ori = pd.to_datetime(df.iloc[:, 0]).tolist()\n",
    "for i in range(test_size):\n",
    "    date_ori.append(date_ori[-1] + timedelta(days = 1))\n",
    "date_ori = pd.Series(date_ori).dt.strftime(date_format = '%Y-%m-%d').tolist()\n",
    "date_ori[-5:]"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Sanity check\n",
    "\n",
    "Some of our models might not have stable gradient, so forecasted trend might really hangwired. You can use many methods to filter out unstable models.\n",
    "\n",
    "This method is very simple,\n",
    "1. If one of element in forecasted trend lower than min(original trend).\n",
    "2. If one of element in forecasted trend bigger than max(original trend) * 2.\n",
    "\n",
    "If both are true, reject that trend."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 13,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "6"
      ]
     },
     "execution_count": 13,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "accepted_results = []\n",
    "for r in results:\n",
    "    if (np.array(r[-test_size:]) < np.min(df['Close'])).sum() == 0 and \\\n",
    "    (np.array(r[-test_size:]) > np.max(df['Close']) * 2).sum() == 0:\n",
    "        accepted_results.append(r)\n",
    "len(accepted_results)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 14,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 1080x360 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "accuracies = [calculate_accuracy(df['Close'].values, r[:-test_size]) for r in accepted_results]\n",
    "\n",
    "plt.figure(figsize = (15, 5))\n",
    "for no, r in enumerate(accepted_results):\n",
    "    plt.plot(r, label = 'forecast %d'%(no + 1))\n",
    "plt.plot(df['Close'], label = 'true trend', c = 'black')\n",
    "plt.legend()\n",
    "plt.title('average accuracy: %.4f'%(np.mean(accuracies)))\n",
    "\n",
    "x_range_future = np.arange(len(results[0]))\n",
    "plt.xticks(x_range_future[::30], date_ori[::30])\n",
    "\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.6.8"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 2
}