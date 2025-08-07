import os
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
print("LSTM imported successfully!")
from keras.models import load_model
from keras.layers import Flatten
from keras.layers import Conv1D
#from keras.layers.convolutional import Conv1D
from keras.layers import MaxPooling1D
#from keras.layers.convolutional import MaxPooling1D
import numpy as np
import time

#from keras.layers import Bidirectional
#from keras.layers import RepeatVector
#from keras.layers import TimeDistributed

from sklearn.metrics import mean_squared_error



def MODEL_LSTM(x_train, x_test, y_train, y_test, desc, train_params, evaler, log_print):    
    '''
    Trains and evaluates an LSTM using data in x_train and y_train
    
    Args:
        x_train:
        x_test:
        y_train:
        y_test:
        train_params: dictionary containing parameters for training the model
        evaler: Evaluator object for model evaluation
    '''
    
    num_exp = train_params['num_exp']
    hidden = train_params['hidden']
    epochs = train_params['epochs']
    forecast_dir = train_params['forecast_dir']
    n_steps_out = train_params['n_steps_out']
    n_steps_in = train_params['n_steps_in']
    n_fvars = train_params['n_fvars']
    
    if forecast_dir and os.path.exists(forecast_dir.joinpath(f"LSTM_{desc}.keras")):
        model_path = forecast_dir.joinpath(f"LSTM_{desc}.keras")
        model = load_model(model_path)
        y_predicttrain = model.predict(x_train)
        y_predicttest = model.predict(x_test)
        evaler.evaluateMetrics(1,desc,y_predicttrain,y_train,y_predicttest,y_test,None)
        best_case_cw = evaler.getMetricScore(1,desc,'CaseWeight')
        best_predict_test = y_predicttest
        best_predict_train = y_predicttrain
        best_model = model
        log_print(f"Loaded forecasting model from {model_path}")
    else:
        model = Sequential()
        model.add(LSTM(hidden, activation='relu', input_shape=(n_steps_in,n_fvars)))
        model.add(Dense(n_steps_out))
        model.compile(optimizer='adam', loss='mse')
        model.summary()
        best_case_cw=1000   #Assigning a large number 
        start_time=time.time()
        log_print(f"Starting LSTM forecaster training on {desc}")
        for run in range(1,num_exp+1):
            log_print(f"\tExperiment {run} in progress")
            # fit model
            start_fit = time.time()
            model.fit(x_train, y_train, epochs=epochs,batch_size=64, verbose=0, shuffle=False)
            fit_time = time.time() - start_fit
            y_predicttrain = model.predict(x_train)
            y_predicttest = model.predict(x_test)
            #evaluate results
            evaler.evaluateMetrics(run,desc,y_predicttrain,y_train,y_predicttest,y_test, fit_time)
            curr_test_case = evaler.getMetricScore(run,desc,'CaseWeight')
            if np.mean(curr_test_case) < np.mean(best_case_cw):
                best_case_cw = curr_test_case
                best_predict_test = y_predicttest
                best_predict_train = y_predicttrain
                best_model = model
        
        log_print(f"Total time for {num_exp} {desc} experiments {time.time()-start_time}")
    
    return best_predict_test, best_predict_train, best_case_cw, best_model

def MODEL_CNN(x_train, x_test, y_train, y_test, desc, train_params, evaler, log_print):
    num_exp = train_params['num_exp']
    hidden = train_params['hidden']
    epochs = train_params['epochs']
    forecast_dir = train_params['forecast_dir']
    n_steps_out = train_params['n_steps_out']
    n_steps_in = train_params['n_steps_in']
    n_fvars = train_params['n_fvars']

    #TODO: load pre-trained forecasting model
    #if forecast_dir and os.path.exists(forecast_dir.joinpath(f"LSTM_{desc}.keras")):

    model = Sequential()
    model.add(Conv1D(filters=32, kernel_size=2, activation='relu', input_shape=(n_steps_in,n_fvars)))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Flatten())
    model.add(Dense(50, activation='relu'))
    model.add(Dense(n_steps_out))
    model.compile(optimizer='adam', loss='mse')
    model.summary()
    best_case_cw=1000   #Assigning a large number 
    start_time=time.time()
    log_print(f"Starting CNN forecaster training on {desc}")
    for run in range(1,num_exp+1):
        log_print(f"\tExperiment {run} in progress")
        # fit model
        start_fit = time.time()
        model.fit(x_train, y_train, epochs=epochs,batch_size=64, verbose=0, shuffle=False)
        fit_time = time.time() - start_fit
        y_predicttrain = model.predict(x_train)
        y_predicttest = model.predict(x_test)
        #evaluate results
        evaler.evaluateMetrics(run,desc,y_predicttrain,y_train,y_predicttest,y_test, fit_time)
        curr_test_case = evaler.getMetricScore(run,desc,'CaseWeight')
        if np.mean(curr_test_case) < np.mean(best_case_cw):
            best_case_cw = curr_test_case
            best_predict_test = y_predicttest
            best_predict_train = y_predicttrain
            best_model = model
    log_print(f"Total time for {num_exp} {desc} experiments {time.time()-start_time}")
    
    return best_predict_test, best_predict_train, best_case_cw, best_model

    
#OLD:
def COMBINED_LSTM(x_train,x_res,x_test,y_train,y_res,y_test,Num_Exp,n_steps_in,n_steps_out,Epochs,Hidden):
    n_features = 1
    x_train = x_train.reshape((x_train.shape[0], x_train.shape[1], n_features))
    print(x_train.shape)
    
    x_res = x_res.reshape((x_res.shape[0], x_res.shape[1], n_features))
    print(x_res.shape)
    
    x_test = x_test.reshape((x_test.shape[0], x_test.shape[1], n_features))
    print(x_test.shape)
    
    train_acc=np.zeros(Num_Exp)
    test_acc=np.zeros(Num_Exp)
    Step_RMSE=np.zeros([Num_Exp,n_steps_out])
    
    model = Sequential()
    model.add(LSTM(Hidden, activation='relu', input_shape=(n_steps_in,n_features)))
    model.add(Dense(n_steps_out))
    model.compile(optimizer='adam', loss='mse')
    model.summary()
    Best_RMSE=1000   #Assigning a large number 
    
    start_time=time.time()
    for run in range(Num_Exp):
        print("Experiment",run+1,"in progress")
        # fit model
        model.fit(x_train, y_train, epochs=Epochs,batch_size=64, verbose=0, shuffle=False)
        y_predicttest1 = model.predict(x_test)
        
        model.fit(x_res, y_res, epochs=Epochs,batch_size=64, verbose=0, shuffle=False)
        y_predicttest2 = model.predict(x_test)
        
        y_combined = y_predicttest2
        for i in range(0,np.shape(y_predicttest1)[0]):
            if(y_predicttest2[i][0] < 0.4):
                y_combined[i][0] = y_predicttest1[i][0]
        test_acc[run] = mean_squared_error( y_combined, y_test) 
        if test_acc[run]<Best_RMSE:
            Best_RMSE=test_acc[run]
            Best_Predict_Test=y_combined
            
    print("Total time for",Num_Exp,"experiments",time.time()-start_time)
    return test_acc,Step_RMSE,Best_Predict_Test


from keras.models import Sequential, load_model
from keras.layers import ConvLSTM2D, Dense, Flatten
import numpy as np
import time
import os
from pathlib import Path



def MODEL_CONV_LSTM(x_train, x_test, y_train, y_test, desc, train_params, evaler, log_print):
    '''
    Trains and evaluates a Conv-LSTM using x_train and y_train
    '''

    num_exp = train_params['num_exp']
    hidden = train_params['hidden']
    epochs = train_params['epochs']
    forecast_dir = train_params['forecast_dir']
    n_steps_out = train_params['n_steps_out']
    n_steps_in = train_params['n_steps_in']
    n_fvars = train_params['n_fvars']

    # **✅ 关键：调整输入形状**（Conv-LSTM 需要 5D 输入）
    x_train = x_train.reshape((x_train.shape[0], x_train.shape[1], 1, 1, x_train.shape[2]))
    x_test = x_test.reshape((x_test.shape[0], x_test.shape[1], 1, 1, x_test.shape[2]))

    # # 确保 forecast_dir 是 Path 对象
    # if isinstance(forecast_dir, str):
    #     forecast_dir = Path(forecast_dir)
    # # 构造模型路径
    # model_path = forecast_dir / f"CONV_LSTM_{desc}.keras"
    # # 检查路径是否存在，然后加载模型
    # if forecast_dir and model_path.exists():
    #     model = load_model(model_path)
    #     log_print(f"Loaded forecasting model from {model_path}")
    if forecast_dir and os.path.exists(forecast_dir.joinpath(f"CONV_LSTM_{desc}.keras")):
        model_path = forecast_dir.joinpath(f"CONV_LSTM_{desc}.keras")
        model = load_model(model_path)
        log_print(f"Loaded forecasting model from {model_path}")
    else:
        model = Sequential()
        model.add(ConvLSTM2D(filters=64, kernel_size=(1, 1), activation='relu', return_sequences=True,
                             input_shape=(n_steps_in, 1, 1, n_fvars)))
        model.add(Flatten())
        model.add(Dense(n_steps_out, activation='linear'))
        model.compile(optimizer='adam', loss='mse')
        model.summary()

        best_case_cw = float('inf')
        start_time = time.time()
        log_print(f"Starting Conv-LSTM forecaster training on {desc}")
        for run in range(1, num_exp + 1):
            log_print(f"\tExperiment {run} in progress")
            start_fit = time.time()
            model.fit(x_train, y_train, epochs=epochs, batch_size=64, verbose=0, shuffle=False)
            fit_time = time.time() - start_fit

            y_predicttrain = model.predict(x_train)
            y_predicttest = model.predict(x_test)

            evaler.evaluateMetrics(run, desc, y_predicttrain, y_train, y_predicttest, y_test, fit_time)
            curr_test_case = evaler.getMetricScore(run, desc, 'CaseWeight')
            if np.mean(curr_test_case) < np.mean(best_case_cw):
                best_case_cw = curr_test_case
                best_predict_test = y_predicttest
                best_predict_train = y_predicttrain
                best_model = model

        log_print(f"Total time for {num_exp} {desc} experiments {time.time() - start_time}")
    print(f"Predictions from Conv-LSTM: {best_predict_test}")

    return best_predict_test, best_predict_train, best_case_cw, best_model


# quantile loss版本的conv-LSTM
from keras.models import Sequential
from keras.layers import ConvLSTM2D, Flatten, Dense
from keras.optimizers import Adam
import numpy as np
import os
import time
from keras.losses import Loss
import tensorflow as tf

# # ✅ 自定义 Quantile Loss 类
# class QuantileLoss(Loss):
#     def __init__(self, quantiles):
#         super().__init__()
#         self.quantiles = quantiles
#
#     def call(self, y_true, y_pred):
#         loss = 0
#         for i, q in enumerate(self.quantiles):
#             e = y_true - y_pred[:, :, i]
#             loss += tf.reduce_mean(tf.maximum(q * e, (q - 1) * e))
#         return loss / len(self.quantiles)
# # utils/losses.py
import tensorflow as tf

class QuantileLoss(tf.keras.losses.Loss):
    def __init__(self, quantiles):
        super().__init__()
        self.quantiles = tf.constant(quantiles, dtype=tf.float32)

    def call(self, y_true, y_pred):
        y_true = tf.expand_dims(y_true, axis=-1)  # [batch, steps, 1]
        e = y_true - y_pred                        # [batch, steps, quantiles]
        loss = tf.maximum(self.quantiles * e, (self.quantiles - 1) * e)
        return tf.reduce_mean(loss)


# ✅ Quantile 版本的 Conv-LSTM 模型函数
def quantile_MODEL_CONV_LSTM(x_train, x_test, y_train, y_test, desc, train_params, evaler, log_print):
    num_exp = train_params['num_exp']
    hidden = train_params['hidden']
    epochs = train_params['epochs']
    forecast_dir = train_params['forecast_dir']
    n_steps_out = train_params['n_steps_out']
    n_steps_in = train_params['n_steps_in']
    n_fvars = train_params['n_fvars']
    quantiles = train_params.get('quantiles', [0.05, 0.5, 0.95])
    n_quantiles = len(quantiles)

    # ✅ reshape input for ConvLSTM2D (5D input)
    x_train = x_train.reshape((x_train.shape[0], n_steps_in, 1, 1, n_fvars))
    x_test = x_test.reshape((x_test.shape[0], n_steps_in, 1, 1, n_fvars))

    model_name = f"QUANTILE_CONV_LSTM_{desc}.keras"
    if forecast_dir and os.path.exists(forecast_dir.joinpath(model_name)):
        model = load_model(forecast_dir.joinpath(model_name), custom_objects={'QuantileLoss': QuantileLoss(quantiles)})
        log_print(f"Loaded forecasting model from {forecast_dir.joinpath(model_name)}")
    else:
        model = Sequential()
        model.add(ConvLSTM2D(filters=64, kernel_size=(1, 1), activation='relu', return_sequences=True,
                             input_shape=(n_steps_in, 1, 1, n_fvars)))
        model.add(Flatten())
        model.add(Dense(n_steps_out * n_quantiles, activation='linear'))
        model.add(tf.keras.layers.Reshape((n_steps_out, n_quantiles)))
        model.compile(optimizer=Adam(), loss=QuantileLoss(quantiles))
        model.summary()

        best_case_cw = float('inf')
        start_time = time.time()
        log_print(f"Starting Quantile Conv-LSTM forecaster training on {desc}")
        for run in range(1, num_exp + 1):
            log_print(f"\tQuantile Experiment {run} in progress")
            start_fit = time.time()
            model.fit(x_train, y_train, epochs=epochs, batch_size=64, verbose=0, shuffle=False)
            fit_time = time.time() - start_fit

            y_predicttrain = model.predict(x_train)
            y_predicttest = model.predict(x_test)

            # ✅ 注意：评估只使用中位数（quantile=0.5）作为评估指标
            y_train_median = y_predicttrain[:, :, quantiles.index(0.5)]
            y_test_median = y_predicttest[:, :, quantiles.index(0.5)]

            evaler.evaluateMetrics(run, desc, y_train_median, y_train, y_test_median, y_test, fit_time)
            curr_test_case = evaler.getMetricScore(run, desc, 'CaseWeight')
            if np.mean(curr_test_case) < np.mean(best_case_cw):
                best_case_cw = curr_test_case
                best_predict_test = y_predicttest  # ← 保留完整预测，包括[0.05, 0.5, 0.95]
                best_predict_train = y_predicttrain
                best_model = model

        log_print(f"Total time for {num_exp} {desc} quantile experiments {time.time() - start_time}")
    print(f"Quantile Predictions from Conv-LSTM: {best_predict_test}")

    return best_predict_test, best_predict_train, best_case_cw, best_model













from keras.layers import LSTM, Bidirectional


def MODEL_BD_LSTM(x_train, x_test, y_train, y_test, desc, train_params, evaler, log_print):
    '''
    Trains and evaluates a Bi-Directional LSTM using x_train and y_train
    '''

    num_exp = train_params['num_exp']
    hidden = train_params['hidden']
    epochs = train_params['epochs']
    forecast_dir = train_params['forecast_dir']
    n_steps_out = train_params['n_steps_out']
    n_steps_in = train_params['n_steps_in']
    n_fvars = train_params['n_fvars']
    print("====== Running BD-LSTM Model ======")

    if forecast_dir and os.path.exists(forecast_dir.joinpath(f"BD_LSTM_{desc}.keras")):
        model_path = forecast_dir.joinpath(f"BD_LSTM_{desc}.keras")
        model = load_model(model_path)
        log_print(f"Loaded forecasting model from {model_path}")
    else:
        model = Sequential()
        model.add(
            Bidirectional(LSTM(hidden, activation='relu', return_sequences=True), input_shape=(n_steps_in, n_fvars)))
        model.add(Bidirectional(LSTM(hidden, activation='relu')))
        model.add(Dense(n_steps_out))
        model.compile(optimizer='adam', loss='mse')
        model.summary()

        best_case_cw = float('inf')
        start_time = time.time()
        log_print(f"Starting BD-LSTM forecaster training on {desc}")
        for run in range(1, num_exp + 1):
            log_print(f"\tExperiment {run} in progress")
            start_fit = time.time()
            model.fit(x_train, y_train, epochs=epochs, batch_size=64, verbose=0, shuffle=False)
            fit_time = time.time() - start_fit

            y_predicttrain = model.predict(x_train)
            y_predicttest = model.predict(x_test)

            evaler.evaluateMetrics(run, desc, y_predicttrain, y_train, y_predicttest, y_test, fit_time)
            curr_test_case = evaler.getMetricScore(run, desc, 'CaseWeight')
            if np.mean(curr_test_case) < np.mean(best_case_cw):
                best_case_cw = curr_test_case
                best_predict_test = y_predicttest
                best_predict_train = y_predicttrain
                best_model = model

        log_print(f"Total time for {num_exp} {desc} experiments {time.time() - start_time}")
    print(f"Predictions from BD-LSTM: {best_predict_test}")
    return best_predict_test, best_predict_train, best_case_cw, best_model


from keras.models import Sequential, load_model
from keras.layers import LSTM, Bidirectional, Dense, Reshape
import tensorflow as tf
import time
import numpy as np
import os

#
# import tensorflow as tf
#
# class QuantileLoss(tf.keras.losses.Loss):
#     def __init__(self, quantiles):
#         super().__init__()
#         self.quantiles = tf.constant(quantiles, dtype=tf.float32)
#
#     def call(self, y_true, y_pred):
#         y_true = tf.expand_dims(y_true, axis=-1)  # [batch, steps, 1]
#         e = y_true - y_pred                        # [batch, steps, quantiles]
#         loss = tf.maximum(self.quantiles * e, (self.quantiles - 1) * e)
#         return tf.reduce_mean(loss)


def quantile_MODEL_BD_LSTM(x_train, x_test, y_train, y_test, desc, train_params, evaler, log_print):
    '''
    Trains and evaluates a Bi-Directional LSTM using Quantile Loss
    '''

    num_exp = train_params['num_exp']
    hidden = train_params['hidden']
    epochs = train_params['epochs']
    forecast_dir = train_params['forecast_dir']
    n_steps_out = train_params['n_steps_out']
    n_steps_in = train_params['n_steps_in']
    n_fvars = train_params['n_fvars']
    quantiles = train_params.get('quantiles', [0.05, 0.5, 0.95])
    n_quantiles = len(quantiles)

    print("====== Running Quantile BD-LSTM Model ======")

    if forecast_dir and os.path.exists(forecast_dir.joinpath(f"QUANTILE_BD_LSTM_{desc}.keras")):
        model_path = forecast_dir.joinpath(f"QUANTILE_BD_LSTM_{desc}.keras")
        model = load_model(model_path)
        log_print(f"Loaded quantile BD-LSTM model from {model_path}")
    else:
        model = Sequential()
        model.add(
            Bidirectional(LSTM(hidden, activation='relu', return_sequences=True), input_shape=(n_steps_in, n_fvars)))
        model.add(Bidirectional(LSTM(hidden, activation='relu')))
        model.add(Dense(n_steps_out * n_quantiles, activation='linear'))
        model.add(tf.keras.layers.Reshape((n_steps_out, n_quantiles)))
        model.compile(optimizer='adam', loss=QuantileLoss(quantiles))
        model.summary()

        best_case_cw = float('inf')
        start_time = time.time()
        log_print(f"Starting Quantile BD-LSTM forecaster training on {desc}")
        for run in range(1, num_exp + 1):
            log_print(f"\tExperiment {run} in progress")
            start_fit = time.time()
            model.fit(x_train, y_train, epochs=epochs, batch_size=64, verbose=0, shuffle=False)
            fit_time = time.time() - start_fit

            y_predicttrain = model.predict(x_train)  # shape: [samples, steps, quantiles]
            y_predicttest = model.predict(x_test)

            # 仅用中位数进行评估
            q_idx = quantiles.index(0.5)
            y_train_median = y_predicttrain[:, :, q_idx]
            y_test_median = y_predicttest[:, :, q_idx]

            evaler.evaluateMetrics(run, desc, y_train_median, y_train, y_test_median, y_test, fit_time)
            curr_test_case = evaler.getMetricScore(run, desc, 'CaseWeight')

            if np.mean(curr_test_case) < np.mean(best_case_cw):
                best_case_cw = curr_test_case
                best_predict_test = y_predicttest
                best_predict_train = y_predicttrain
                best_model = model

        log_print(f"Total time for {num_exp} {desc} experiments {time.time() - start_time}")

    print(f"Predictions from Quantile BD-LSTM: shape {best_predict_test.shape}")
    return best_predict_test, best_predict_train, best_case_cw, best_model

import numpy as np
from tensorflow.keras import layers, models
from tensorflow.keras.losses import MeanSquaredError
import tensorflow as tf

# ---------- Quantile Loss ----------
def quantile_loss(q):
    def loss(y_true, y_pred):
        e = y_true - y_pred
        return tf.reduce_mean(tf.maximum(q * e, (q - 1.) * e))
    return loss

# ---------- 构建单个 ConvLSTM 模型 ----------
def _build_convlstm(input_shape, output_dim, hidden, loss_fn):
    model = models.Sequential()
    model.add(layers.ConvLSTM2D(filters=hidden,
                                 kernel_size=(1, 1),
                                 activation='relu',
                                 return_sequences=False,
                                 input_shape=input_shape))
    model.add(layers.Flatten())
    model.add(layers.Dense(output_dim))
    model.compile(optimizer='adam', loss=loss_fn)
    return model

# ---------- 主函数：Quantile Conv-LSTM ----------
def MODEL_QUANTILE_CONV_LSTM(x_train, x_test, y_train, y_test, desc, train_params, evaler, log_print):
    '''
    Trains and evaluates a Quantile Conv-LSTM Ensemble model using x/y data
    '''
    num_exp = train_params['num_exp']
    hidden = train_params['hidden']
    epochs = train_params['epochs']
    n_steps_out = train_params['n_steps_out']
    n_steps_in = train_params['n_steps_in']
    n_fvars = train_params['n_fvars']

    # ✅ reshape for ConvLSTM: (samples, time, rows, cols, features)
    x_train = x_train.reshape((x_train.shape[0], x_train.shape[1], 1, 1, x_train.shape[2]))
    x_test  = x_test.reshape((x_test.shape[0],  x_test.shape[1],  1, 1, x_test.shape[2]))

    input_shape = (x_train.shape[1], 1, 1, x_train.shape[4])  # (timesteps, 1, 1, features)
    output_dim = n_steps_out

    # 创建三个模型
    model_q50 = _build_convlstm(input_shape, output_dim, hidden, MeanSquaredError())
    model_q05 = _build_convlstm(input_shape, output_dim, hidden, quantile_loss(0.05))
    model_q95 = _build_convlstm(input_shape, output_dim, hidden, quantile_loss(0.95))

    best_cw = float('inf')
    best_model = None
    best_predict_test = None

    log_print(f"====== Running Quantile Conv-LSTM [{desc}] ======")

    for run in range(1, num_exp + 1):
        log_print(f"\tExperiment {run} in progress")

        model_q50.fit(x_train, y_train, epochs=epochs, batch_size=64, verbose=0, shuffle=False)
        model_q05.fit(x_train, y_train, epochs=epochs, batch_size=64, verbose=0, shuffle=False)
        model_q95.fit(x_train, y_train, epochs=epochs, batch_size=64, verbose=0, shuffle=False)

        # pred_q50 = model_q50.predict(x_test, verbose=0)
        # rmse = evaler.evaluate_RMSE(y_test, pred_q50)     # ✅正确方式
        #
        #
        # if rmse < best_cw:
        #     best_cw = rmse
        #     best_predict_test = pred_q50
        #     best_model = {
        #         "q50": model_q50,
        #         "q05": model_q05,
        #         "q95": model_q95
        #     }
        import time

        start_fit = time.time()

        model_q50.fit(x_train, y_train, epochs=epochs, batch_size=64,
                      verbose=0, shuffle=False)
        model_q05.fit(x_train, y_train, epochs=epochs, batch_size=64,
                      verbose=0, shuffle=False)
        model_q95.fit(x_train, y_train, epochs=epochs, batch_size=64,
                      verbose=0, shuffle=False)

        fit_time = time.time() - start_fit

        # 预测
        y_predicttrain = model_q50.predict(x_train, verbose=0)
        y_predicttest = model_q50.predict(x_test, verbose=0)

        # 统一评估与获取指标
        run_id = 1
        for tag, mdl in {"q05": model_q05,
                         "q50": model_q50,
                         "q95": model_q95}.items():
            y_pred_train = mdl.predict(x_train, verbose=0)
            y_pred_test = mdl.predict(x_test, verbose=0)

            # desc 里追加分位标签，例：no_resample_q05
            evaler.evaluateMetrics(run_id, f"{desc}_{tag}",
                                   y_pred_train, y_train,
                                   y_pred_test, y_test,
                                   fit_time)

        # 仍以 q50 作为“最佳点预测”选 CaseWeight
        best_cw = evaler.getMetricScore(run_id, f"{desc}_q50", 'CaseWeight')
        best_predict_test = y_predicttest
        best_predict_train = y_predicttrain
        best_model = {
            "q50": model_q50,
            "q05": model_q05,
            "q95": model_q95
        }

    return best_predict_test, None, best_cw, best_model

