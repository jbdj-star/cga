from __future__ import print_function
import os
import sys
import pickle
import time
import matplotlib.pyplot as plt
import numpy as np
import h5py
import flowdatacheck_taxi
from keras.layers import (
    Input,
    Activation,
    merge,
    Dense,
    Reshape,
    Flatten,
    add,
    multiply,
GlobalAveragePooling2D
)
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers.convolutional import Conv2D, UpSampling2D, Conv2DTranspose
from keras.layers.normalization import BatchNormalization
from keras.layers.pooling import MaxPooling2D
from keras.models import Model
import numpy as np
from keras.optimizers import RMSprop, Adam

np.random.seed(1337)

X_train, Y_train, X_test, Y_test = flowdatacheck_taxi.X_train, flowdatacheck_taxi.Y_train, flowdatacheck_taxi.X_test, flowdatacheck_taxi.Y_test



def _shortcut(input, residual):
    return merge([input, residual], mode='sum')


def _bn_relu_conv(nb_filter, nb_row, nb_col, subsample=(1, 1), bn=False):
    def f(input):
        if bn:
            input = BatchNormalization(mode=0, axis=1)(input)
        activation = Activation('relu')(input)
        return Conv2D(64, (3, 3), strides=(1, 1), padding="same")(activation)
    return f


def _residual_unit(nb_filter, init_subsample=(1, 1)):
    def f(input):
        residual = _bn_relu_conv(nb_filter, 3, 3, subsample=init_subsample)(input)
        residual = _bn_relu_conv(nb_filter, 3, 3, subsample=init_subsample)(residual)
        return _shortcut(input, residual)
    return f

def SE_block_T(len_seq):
    def f(input):
        excitation = Flatten()(input)
        #excitation = GlobalAveragePooling2D()(input)
        excitation = Dense(output_dim=64, activation='relu')(excitation)
        excitation = Dense(output_dim=10 * 20 * len_seq*2, activation='sigmoid')(excitation)
        excitation = Reshape((len_seq*2, 10, 20))(excitation)
        return excitation
    return f

def SE_block():
    def f(input):
        excitation = Flatten()(input)
        excitation = Dense(output_dim=64, activation='relu')(excitation)
        excitation = Dense(output_dim=10 * 20 * 64, activation='sigmoid')(excitation)
        excitation = Reshape((64, 10, 20))(excitation)
        return excitation
    return f

def down_up_block():
    def f(input):
        input = Conv2D(64, kernel_size=(3, 3), strides=(1, 1), activation='relu', padding='same')(input)
        #input = MaxPooling2D(pool_size=(2, 2), strides=(1, 1), padding='same')(input)

        #input = MaxPooling2D(pool_size=(2, 2), strides=(1, 1), padding='same')(input)
        input = Conv2D(64, kernel_size=(3, 3), strides=(1, 1), activation='sigmoid', padding='same')(input)
        input = Conv2DTranspose(64, kernel_size=(3, 3), strides=(1, 1), activation='sigmoid', padding='same')(input)

        input = Conv2DTranspose(64, kernel_size=(3, 3), strides=(1, 1), activation='sigmoid', padding='same')(input)
        return input
    return f


def ResUnits(nb_filter, block_repetations=1):
    def f(input):
        for i in range(block_repetations):
            init_subsample = (1, 1)
            input = _residual_unit(nb_filter=nb_filter,
                                  init_subsample=init_subsample)(input)
        return input
    return f

def AttenResUnits(nb_filter, block_repetations=1):
    def f(input):
        for i in range(block_repetations):
            #down_up_input = down_up_block()(input)
            SE_block_input = SE_block()(input)
            init_subsample = (1, 1)
            input = _residual_unit(nb_filter=nb_filter,
                                  init_subsample=init_subsample)(input)
            #M_x = merge([down_up_input, input], mode='mul')
            M_x = merge([SE_block_input, input], mode='mul')
        return M_x
    return f


def stresnet(c_conf=(4, 2, 10, 20), p_conf=(3, 2, 10, 20), t_conf=(1, 2, 10, 20), nb_residual_unit=1):
    '''
    C - Temporal Closeness
    P - Period
    T - Trend
    conf = (len_seq, nb_flow, map_height, map_width)
    external_dim
    '''

    # main input
    main_inputs = []
    outputs = []
    for conf in [c_conf, p_conf, t_conf]:
        if conf is not None:
            len_seq, nb_flow, map_height, map_width = conf
            input = Input(shape=(nb_flow * len_seq, map_height, map_width))
            main_inputs.append(input)

            if len_seq != 1:
                SE_block_t = SE_block_T(len_seq=len_seq)(input)
                input = merge([SE_block_t, input], mode='mul')

            # Conv1
            conv1 = Conv2D(64, kernel_size=(3, 3), strides=(1, 1), padding="same")(input)
            # [nb_residual_unit] Residual Units
            #residual_output = ResUnits(nb_filter=64, block_repetations=nb_residual_unit)(conv1)
            residual_output = AttenResUnits(nb_filter=64, block_repetations=nb_residual_unit)(conv1)
            #residual_output = SE_ResUnits(block_repetations=nb_residual_unit)(conv1)
            # Conv2
            activation = Activation('relu')(residual_output)
            conv2 = Conv2D(64, kernel_size=(3, 3), strides=(1, 1), padding="same")(activation)
            outputs.append(conv2)

    # parameter-matrix-based fusion
    if len(outputs) == 1:
        main_output = outputs[0]
    else:
        import iLayer
        new_outputs = []
        for output in outputs:
            new_outputs.append(iLayer.iLayer()(output))
        main_output = merge(outputs, mode='sum')

    conv3 = Conv2D(2, kernel_size=(3, 3), strides=(1, 1), padding="same")(main_output)

    main_output = Activation('sigmoid')(conv3)
    model = Model(input=main_inputs, output=main_output)

    return model

def rmse(Y_true, Y_pred):
    # https://www.kaggle.com/wiki/RootMeanSquaredError
    from sklearn.metrics import mean_squared_error
    print("===RMSE===")
    # in
    RMSE = mean_squared_error(Y_true[:, 0].flatten(), Y_pred[:, 0].flatten())**0.5
    print('inflow: ', RMSE)
    print(RMSE*1289)
    # out
    if Y_true.shape[1] > 1:
        RMSE = mean_squared_error(Y_true[:, 1].flatten(), Y_pred[:, 1].flatten())**0.5
        print('outflow: ', RMSE)
        print(RMSE * 1289)

    RMSE = mean_squared_error(Y_true.flatten(), Y_pred.flatten())**0.5

    print("total rmse: ", RMSE)
    print(RMSE * 1289)

    print("===RMSE===")
    return RMSE

def mean_absolute_percentage_error(y_true, y_pred):
    idx = np.nonzero(y_true)
    return np.mean(np.abs((y_true[idx] - y_pred[idx]) / y_true[idx])) * 100


def eval_threshold(y, pred_y, threshold=0.007758):
    pickup_y = y[:, 0]
    dropoff_y = y[:, 1]
    pickup_pred_y = pred_y[:, 0]
    dropoff_pred_y = pred_y[:, 1]
    pickup_mask = pickup_y > threshold
    dropoff_mask = dropoff_y > threshold
    print("===eval_threshold===")
    #pickup part
    if np.sum(pickup_mask)!=0:
        avg_pickup_mape = np.mean(np.abs(pickup_y[pickup_mask]-pickup_pred_y[pickup_mask])/pickup_y[pickup_mask])
        avg_pickup_rmse = np.sqrt(np.mean(np.square(pickup_y[pickup_mask]-pickup_pred_y[pickup_mask])))
        print('inflowRMSE: ', avg_pickup_rmse)
        print(avg_pickup_rmse*1289)
        print('inflowMAPE: ', avg_pickup_mape)
    #dropoff part
    if np.sum(dropoff_mask)!=0:
        avg_dropoff_mape = np.mean(np.abs(dropoff_y[dropoff_mask]-dropoff_pred_y[dropoff_mask])/dropoff_y[dropoff_mask])
        avg_dropoff_rmse = np.sqrt(np.mean(np.square(dropoff_y[dropoff_mask]-dropoff_pred_y[dropoff_mask])))
        print('outflowRMSE: ', avg_dropoff_rmse)
        print(avg_dropoff_rmse * 1289)
        print('outflowMAPE: ', avg_dropoff_mape)

    return (avg_pickup_rmse, avg_pickup_mape), (avg_dropoff_rmse, avg_dropoff_mape)


def eval_threshold_all(y, pred_y, threshold=0.007758):
    pickup_y = y

    pickup_pred_y = pred_y

    pickup_mask = pickup_y > threshold

    print("===eval_threshold_all===")
    #pickup part
    if np.sum(pickup_mask)!=0:
        avg_pickup_mape = np.mean(np.abs(pickup_y[pickup_mask]-pickup_pred_y[pickup_mask])/pickup_y[pickup_mask])
        avg_pickup_rmse = np.sqrt(np.mean(np.square(pickup_y[pickup_mask]-pickup_pred_y[pickup_mask])))
        print('allRMSE: ', avg_pickup_rmse)
        print(avg_pickup_rmse*1289)
        print('allMAPE: ', avg_pickup_mape)
    return (avg_pickup_rmse, avg_pickup_mape)


def mape(Y_true, Y_pred):
    print("===MAPE===")
    # in
    MAPE = mean_absolute_percentage_error(Y_true[:, 0].flatten(), Y_pred[:, 0].flatten())
    print("inflow: ", MAPE)
    # out
    MAPE = mean_absolute_percentage_error(Y_true[:, 1].flatten(), Y_pred[:, 1].flatten())
    print("outflow: ", MAPE)
    MAPE = mean_absolute_percentage_error(Y_true.flatten(), Y_pred.flatten())
    print("total mape: ", MAPE)
    print("===MAPE===")
    return MAPE



def draw(Y_true, Y_pred, flow=0, r=5, c=11, yl=1000):
    y_true1 = Y_true[:, flow, r, c].flatten()
    y_pred1 = Y_pred[:, flow, r, c].flatten()
    x1 = range(len(y_true1))
    x2 = range(len(y_pred1))
    y_true1 = y_true1 * 1289
    y_pred1 = y_pred1 * 1289

    plt.plot(x1, y_true1, color="red", linewidth=1, linestyle="-", label="ground truth")
    plt.plot(x2, y_pred1, color="blue", linewidth=1, linestyle="-", label="prediction")
    plt.xlim(0, len(y_true1))
    plt.ylim(0, yl)
    plt.show()

def drawday(Y_true, Y_pred, flow=0, r=5, c=11, yl=1000, days=10, daye=106):
    y_true1 = Y_true[:, flow, r, c].flatten()
    y_pred1 = Y_pred[:, flow, r, c].flatten()
    x1 = range(len(y_true1))
    x2 = range(len(y_pred1))
    y_true1 = y_true1 * 1289
    y_pred1 = y_pred1 * 1289

    plt.plot(x1, y_true1, color="red", linewidth=1, linestyle="-", label="ground truth")
    plt.plot(x2, y_pred1, color="blue", linewidth=1, linestyle="-", label="prediction")
    plt.xlim(days, daye )
    plt.ylim(0, yl)
    plt.show()



if __name__ == '__main__':
    len_closeness = 4
    len_period = 3
    len_trend = 1
    nb_residual_unit = 1
    lr = 0.0001
    nb_epoch = 500  # number of epoch at training stage
    nb_epoch_cont = 100  # number of epoch at training (cont) stage
    batch_size = 32  # batch size
    T = 48  # number of time intervals in one day
    path_result = 'RET'
    if os.path.isdir(path_result) is False:
        os.mkdir(path_result)
    path_model = 'MODEL'
    if os.path.isdir(path_model) is False:
        os.mkdir(path_model)

    # rmsprop = RMSprop(lr=0.00045, decay=0.0)
    rmsprop = Adam(lr=lr)
    stresnet = stresnet(nb_residual_unit=nb_residual_unit)
    stresnet.compile(optimizer=rmsprop, loss='mse', )
    stresnet.summary()
    hyperparams_name = 'c{}.p{}.t{}.resunit{}.lr{}'.format(
        len_closeness, len_period, len_trend, nb_residual_unit, lr)
    fname_param = os.path.join('MODEL', '{}.best.h5'.format(hyperparams_name))
    early_stopping = EarlyStopping(monitor='val_loss', patience=20, mode='min')
    model_checkpoint = ModelCheckpoint(fname_param, monitor='val_loss', verbose=0, save_best_only=True, mode='min')

    print('=' * 10)
    print("training model...")
    ts = time.time()
    history = stresnet.fit(X_train, Y_train,
                           epochs=nb_epoch,
                           batch_size=batch_size,
                           validation_split=0.1,
                           callbacks=[early_stopping, model_checkpoint],
                           verbose=1)
    stresnet.save_weights(os.path.join(
        'MODEL', '{}.h5'.format(hyperparams_name)), overwrite=True)
    pickle.dump((history.history), open(os.path.join(
        path_result, '{}.history.pkl'.format(hyperparams_name)), 'wb'))
    print("\nelapsed time (training): %.3f seconds\n" % (time.time() - ts))

    print('=' * 10)
    print('evaluating using the model that has the best loss on the valid set')
    ts = time.time()
    stresnet.load_weights(fname_param)
    score = stresnet.evaluate(X_train, Y_train, batch_size=Y_train.shape[
                                                               0], verbose=0)
    print('Train mse : %.6f rmse (real): %.6f' %
          (score, (score ** 0.5) * (1283 - 0)))
    score = stresnet.evaluate(
        X_test, Y_test, batch_size=Y_test.shape[0], verbose=0)
    print('Test mse : %.6f rmse (real): %.6f' %
          (score, (score ** 0.5) * (1289 - 0)))
    print("\nelapsed time (eval): %.3f seconds\n" % (time.time() - ts))

    print('=' * 10)
    print("training model (cont)...")
    ts = time.time()
    fname_param = os.path.join(
        'MODEL', '{}.cont.best.h5'.format(hyperparams_name))
    model_checkpoint = ModelCheckpoint(
        fname_param, monitor='val_loss', verbose=0, save_best_only=True, mode='min')

    history = stresnet.fit(X_train, Y_train, epochs=nb_epoch_cont, verbose=1, batch_size=batch_size,
                           callbacks=[model_checkpoint])

    pickle.dump((history.history), open(os.path.join(
        path_result, '{}.cont.history.pkl'.format(hyperparams_name)), 'wb'))
    stresnet.save_weights(os.path.join(
        'MODEL', '{}_cont.h5'.format(hyperparams_name)), overwrite=True)
    print("\nelapsed time (training cont): %.3f seconds\n" % (time.time() - ts))

    print('=' * 10)
    print('evaluating using the final model')
    score = stresnet.evaluate(X_train, Y_train, batch_size=Y_train.shape[0], verbose=0)
    print('Train mse (norm): %.6f rmse (real): %.6f' %
          (score, (score ** 0.5) * (1283 - 0)))
    ts = time.time()
    score = stresnet.evaluate(X_test, Y_test, batch_size=32, verbose=0)
    print('Test mse (norm): %.6f rmse (real): %.6f' %(score, (score ** 0.5) * (1289 - 0)))
    print("\nelapsed time (eval cont): %.3f seconds\n" % (time.time() - ts))


    print('\nTesting ------------')

