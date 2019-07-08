import tensorflow as tf
from tensorflow.keras import models
from tensorflow.keras import layers
import keras


import math
import numpy as np
import matplotlib.pyplot as plt
import os
import pickle
from sklearn.metrics import mean_squared_error



# TODO:
# Calculate MSE for all considered configurations
# Choose the best network architecture


# to make this notebook's output stable across runs
def reset_graph(seed=42):
    tf.reset_default_graph()
    tf.set_random_seed(seed)
    np.random.seed(seed)


# Where to save the figures
PROJECT_ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

def save_fig(fig_id, tight_layout=True):
    path = os.path.join(PROJECT_ROOT_DIR, "images_testing_rnn", fig_id + ".png")
    print("Saving figure", fig_id)
    # if tight_layout:
    #     plt.tight_layout()
    plt.savefig(path, format='png', dpi=300)


a = 1
b = 0.1


def calculate_inverse_static_characteristic(y):
    return b * y


def simulate_step(control, previous_output, a=1, b=0.1, NIS=10):
    dt = 1/NIS

    y = previous_output
    u = control

    for i in range(0, NIS):
        dy = (1 / a * u - b / a * y) * dt
        # print('\n' + "dy = " + str(dy))
        y = y + dy

    return y


def acquire_data_for_training(start_y, end_y, step=0.01):
    output = range(start_y, step, end_y)
    control = []

    for y in output:
        output.append(calculate_inverse_static_characteristic(y))

    return dict({"control": control, "output": output})


def simulate_inertia(u=1, y=0, time=10):
    for i in range(0, time):
        y = simulate_step(u, y)
        print(str(y) + '\n')


def time_series(t):
    return t * np.sin(t) / 4 + 5 * np.sin(t*3)


def next_batch(experiment_length, control_full, n_steps, previous_output=0,
               output_full=[]): # bylo jeszcze batch_size ale z niego nie korzystam
    # t0 = np.random.rand(batch_size, 1) * (t_max - t_min - n_steps * resolution)  # to produkuje macierz batch_size x n_steps, czyli ileś paczuszek po n_steps wartosci każda
                                                                                 # czyli na przykład ładuje wektor sterowań (albo kilka paczuszek wektorów sterowań), a sieć
                                                                                 # powinna wypluć wyjście (albo wektor wyjść jeżeli wrzucam kilka paczek)
    # Ts = t0 + np.arange(0., n_steps + 1) * resolution
    # ys_ = time_series(Ts)
    #             # paczuszki z obcięttą jedną wartością na końcu | a tu paczuszki przesunięte względem początku sekwencji o 1 (czyli zaczynają się od indeksu 1 a nie od inddeksu 0)
    # ys_1, ys_2 = ys_[:, :-1].reshape(-1, n_steps, 1), ys_[:, 1:].reshape(-1, n_steps, 1)
    # # w szeregah czasowych szuka sie zaleznosci pomiedzy poprzednia a nastepna probka, to tak jakby szukac zaleznosci pomiezy poprzednim wyjsciem a nastepnym wyjsciem obiektu, a to bzdura, bo zalezy od sterowania i/albo od stanu

    y = []
    if len(output_full) == 0:
        for i in range(0, experiment_length):  #było też n_steps + 1 ale to dla błędnej koncepcji :p
            current_output = simulate_step(control_full[i], previous_output=previous_output)
            y.append(current_output)
            previous_output = current_output
    else:
        y = output_full

    ys = np.asarray([np.asarray(y)])

    control_X_batch = []  # np.asarray([control_full]).reshape(-1, n_steps, 1)
    output_y_batch = []  # ys.reshape(-1, n_steps, 1)

    control_X_batch_flat = control_full
    output_y_batch_flat = y

    return control_X_batch, output_y_batch, previous_output, control_X_batch_flat, output_y_batch_flat


def construct_rnn(n_steps, n_inputs, n_outputs, n_neurons):
    # X_batch, y_batch = next_batch(1, n_steps)

    reset_graph()

    X = tf.placeholder(tf.float32, [None, n_steps, n_inputs])
    y = tf.placeholder(tf.float32, [None, n_steps, n_outputs])

    # cell = tf.contrib.rnn.BasicRNNCell(num_units=n_neurons, activation=tf.nn.relu)
    cell = tf.contrib.rnn.OutputProjectionWrapper(
        # tf.keras.layers.SimpleRNNCell(units=n_neurons, activation=tf.nn.relu),
        # output_size=n_outputs)
        tf.nn.rnn_cell.BasicRNNCell(num_units=n_neurons, activation=tf.nn.relu),
        output_size=n_outputs)

    outputs, states = tf.nn.dynamic_rnn(cell, X, dtype=tf.float32)
    # outputs, states = tf.keras.layers.RNN(cell)

    learning_rate = 0.001

    loss = tf.reduce_mean(tf.square(outputs - y))
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    training_op = optimizer.minimize(loss)

    init = tf.global_variables_initializer()

    return init, training_op, X, y, outputs, loss


def run_and_plot_rnn(init, training_op, X, y, outputs, loss, saver, n_iterations, n_steps,
                     control_full, control_full_test, title, experiment_length, output_full=[], output_full_test=[],
                     option='', ploting=False, training_signal_addon="inertia_modelling_checkpoints"):
    with tf.Session() as sess:
        init.run()
        previous_output = 0
        X_batch_full, y_batch_full, previous_output, control_X_batch_flat, output_y_batch_flat = next_batch(
            experiment_length, control_full, n_steps, previous_output, output_full=output_full)  # bylo jeszcze batch
                                                                                                 #_size ale z tego nie korzystam

        control_X_batch_flat = [0] * (n_steps - 1) + control_X_batch_flat
        output_y_batch_flat = [0] * (n_steps - 1) + output_y_batch_flat

        for iteration in range(n_iterations):
            if iteration % 10 == 0:
                if option != '':
                    print("Normal model " + "./active_suspension_modelling_checkpoints/my_time_series_model"
                          + title + " train, iteration:" + str(iteration))
                else:
                    print("Normal model " + "./" + training_signal_addon + "/my_time_series_model"
                          + title + " train, iteration:" + str(iteration))
            for i in range(0, len(control_full) - n_steps):
                temp_y = np.asarray([np.asarray(output_y_batch_flat[i:i + n_steps])]).reshape(-1, n_steps, 1)
                temp_x = np.asarray([np.asarray(control_X_batch_flat[i:i + n_steps])]).reshape(-1, n_steps, 1)
                sess.run(training_op, feed_dict={X: temp_x, y: temp_y})

        output_y_train_batch_flat = output_y_batch_flat

        if option != '':
            saver.save(sess,
                       "./active_suspension_modelling_checkpoints/my_time_series_model" + title)  # not shown in the book
            print("Normal model " + "./active_suspension_modelling_checkpoints/my_time_series_model"
                  + title + " done and saved")
        else:
            saver.save(sess, "./" + training_signal_addon + "/my_time_series_model" + title)  # not shown in the book
            print("Normal model " + "./" + training_signal_addon + "/my_time_series_model"
                  + title + " done and saved")

    with tf.Session() as sess:  # not shown in the book
        if option != '':
            saver.restore(sess, "./active_suspension_modelling_checkpoints/my_time_series_model" + title)
        else:
            saver.restore(sess, "./" + training_signal_addon + "/my_time_series_model" + title)

        # Testowanie na zbiorze uczącym
        output_prediction_train = []
        for i in range(0, len(control_full) - n_steps):
            new_temp_x = np.asarray([np.asarray(control_X_batch_flat[i:i + n_steps])]).reshape(-1, n_steps, 1)
            y_pred = sess.run(outputs, feed_dict={X: new_temp_x})
            output_prediction_train.append(y_pred[-1][-1])

        output_prediction_test = []
        control_full_test = [0] * (n_steps - 1) + control_full_test
        # Testowanie na zbiorze nowym
        for i in range(0, len(control_full_test) - n_steps):
            new_temp_x = np.asarray([np.asarray(control_full_test[i:i + n_steps])]).reshape(-1, n_steps, 1)
            y_pred_test = sess.run(outputs, feed_dict={X: new_temp_x})
            output_prediction_test.append(y_pred_test[-1][-1])

    previous_output = 0
    unused1, y_batch_test, unused2, control_X_batch_flat, output_y_test_batch_flat = next_batch(len(control_full_test),
                                                                                           control_full_test, n_steps,
                                                                                           previous_output,
                                                                                           output_full=output_full_test)
    # ploting test set generated data
    if ploting:
        plt.title(title, fontsize=10)
        plt.plot(output_y_test_batch_flat[n_steps:], "g-", markersize=3,
                 label="instance_test")
        plt.plot(output_prediction_test, "m.", markersize=3,
                 label="prediction_test")
        plt.xlabel("Time")
        plt.title('Model performance Test-set-generated-data')
        plt.legend()
        plt.show()

    try:
        # MSE dla testowego - traning konczy sie w "ucięty sposób" bo ostatnia probka sterowania jaka bierzesz
        # to n_steps od konca
        mse_train_set = mean_squared_error(output_y_train_batch_flat[n_steps-1:-n_steps], output_prediction_train)
    except ValueError:
        mse_train_set = 0.0

    try:
        mse_test_set = mean_squared_error(output_y_test_batch_flat[n_steps:], output_prediction_test)
    except ValueError:
        mse_test_set = 0.0

    if option != '':
        print("Model " + "./active_suspension_modelling_checkpoints/my_time_series_model"
              + title + " tested. MSE calculated")
    else:
        print("Model " + "./" + training_signal_addon + "/my_time_series_model"
              + title + " tested. MSE calculated")

    # save_fig(title)
    return mse_train_set, mse_test_set


def run_and_plot_rnn_inverse(init, training_op, X, y, outputs, loss, saver, n_iterations, n_steps,
                             control_full, control_full_test, title, experiment_length, option='',
                             output_full=[], output_full_test=[], ploting=False,
                             training_signal_addon="inertia_modelling_checkpoints"):
    with tf.Session() as sess:
        init.run()
        previous_output = 0
        X_batch_full, y_batch_full, previous_output, control_X_batch_flat, output_y_batch_flat = next_batch(
            experiment_length, control_full, n_steps, previous_output, output_full=output_full)  # bylo jeszcze batch_size ale z tego nie korzystam

        control_X_batch_flat = [0] * (n_steps - 1) + control_X_batch_flat
        output_y_batch_flat = [0] * (n_steps - 1) + output_y_batch_flat

        if option != '':
            print("Inverse model " + "./active_suspension_modelling_checkpoints/my_time_series_model"
                  + title + " train")
        else:
            print("Inverse model " + "./" + training_signal_addon + "/my_time_series_model"
                  + title + " train")

        for iteration in range(n_iterations):
            for i in range(0, len(control_full) - n_steps):
                temp_y = np.asarray([np.asarray(output_y_batch_flat[i:i + n_steps])]).reshape(-1, n_steps, 1)
                temp_x = np.asarray([np.asarray(control_X_batch_flat[i:i + n_steps])]).reshape(-1, n_steps, 1)
                sess.run(training_op, feed_dict={X: temp_y, y: temp_x})

        if option != '':
            saver.save(sess,
                       "./active_suspension_modelling_checkpoints/my_time_series_model" + title)  # not shown in the book
        else:
            saver.save(sess,
                       "./" + training_signal_addon + "/my_time_series_model" + title)  # not shown in the book

    input_prediction_test = []
    previous_output = 0

    X_batch_full, y_batch_full, previous_output, control_X_batch_flat, output_y_batch_flat_test = next_batch(
        len(control_full_test), control_full_test, n_steps, previous_output, output_full=output_full_test)  # bylo jeszcze batch_size ale z tego nie korzystam

    if option != '':
        print("Inverse model " + "./active_suspension_modelling_checkpoints/my_time_series_model"
              + title + " test")
    else:
        print("Inverse model " + "./" + training_signal_addon + "/my_time_series_model"
              + title + " test")

    with tf.Session() as sess:  # not shown in the book
        if option != '':
            saver.restore(sess, "./active_suspension_modelling_checkpoints/my_time_series_model" + title)  # not shown
        else:
            saver.restore(sess, "./" + training_signal_addon + "/my_time_series_model" + title)  # not shown

        # Testowanie na zbiorze uczącym
        input_prediction_train = []
        for i in range(0, len(control_full) - n_steps):
            new_temp_y = np.asarray([np.asarray(output_y_batch_flat[i:i + n_steps])]).reshape(-1, n_steps, 1)
            X_pred = sess.run(outputs, feed_dict={X: new_temp_y})
            input_prediction_train.append(X_pred[-1][-1])

        # Testowanie na zbiorze testowym
        for i in range(0, len(control_full_test) - n_steps):
            new_temp_y = np.asarray([np.asarray(output_y_batch_flat_test[i:i + n_steps])]).reshape(-1, n_steps, 1)
            X_pred_test = sess.run(outputs, feed_dict={X: new_temp_y})
            input_prediction_test.append(X_pred_test[-1][-1])

    if ploting:
        plt.title(title, fontsize=10)
        plt.plot(control_full_test[n_steps:], "g-", markersize=5,
                 label="instance_test")
        plt.plot(input_prediction_test, "m-", markersize=5,
                 label="prediction_test")
        plt.xlabel("Time")
        plt.title('Inverse model permormance - Test set generated data')
        plt.show()

    # MSE
    try:
        mse_train_set = mean_squared_error(control_full[n_steps:], input_prediction_train)
    except ValueError:
        mse_train_set = 0.0

    try:
        mse_test_set = mean_squared_error(control_full_test[n_steps:], input_prediction_test)
    except ValueError:
        mse_test_set = 0.0

    # save_fig(title)
    return mse_train_set, mse_test_set

# BRAKUJE Z TEJ PRZERWANEJ SYMULACJI TO BRAKUJE TYLKO KONFIGURACJI 10, 50, 1000
def perform_identification(control_full, full_experiment_length, control_full_test, title_addon='',
                           n_iterations_list=[20, 50], #[10, 20, 50],  # czyli ile razy przejeżdżam przez cały zbiór danych
                           # batch_size_list=[2, 25],  # , 100]
                           n_neurons_list=[50, 200, 500, 1000],  # [1, 10, 100]
                           n_steps_list=[10, 20, 30, 50],  # czyli ile poprzednich sterowań biorę pod uwagę
                           mode='basicRNN',
                           option='', output_full=[], output_full_test=[], ploting=False,
                           training_signal_addon="inertia_modelling_checkpoints"):
    n_inputs = 1
    n_outputs = 1

    all_models_counter = len(n_iterations_list) * len(n_neurons_list) * len(n_steps_list)
    all_models_inverse_counter = all_models_counter
    models_counter = 1

    model_performance = []
    model_inverse_performance = []

    for n_iterations in n_iterations_list:
    #    for batch_size in batch_size_list:  # poki co nie korzystam z batch size!!!
        for n_steps in n_steps_list:
            for n_neurons in n_neurons_list:
                title = "_FRAMED_n_iterations_" + str(n_iterations) + " n_steps_" + str(
                    n_steps) + " n_neurons_" + str(n_neurons)  #  + " batch_size_" + str(batch_size) # poki co nie korzystam z batch size
                title = title_addon + title

                init, training_op, X, y, outputs, loss = construct_rnn(n_steps, n_inputs, n_outputs, n_neurons)

                saver = tf.train.   Saver()

                mse = run_and_plot_rnn(init, training_op, X, y, outputs, loss, saver, n_iterations, n_steps,
                                       control_full, control_full_test, title, full_experiment_length,
                                       output_full=output_full, output_full_test=output_full_test,
                                       option=option, ploting=ploting, training_signal_addon=training_signal_addon)
                model_performance.append(
                    {
                        'MSE': mse,
                        'n_steps': n_steps, 'n_neurons': n_neurons, 'n_iterations': n_iterations,
                        'title': title,
                        'identification_option': option, 'training_signal_addon': training_signal_addon
                    })
                print("Model " + str(models_counter) + " of " + str(all_models_counter))
                models_counter = models_counter + 1

    with open(r"model_and_inv_model_identification_mses/model_performance" + title_addon + ".pickle" + option,
              "wb") as output_file:
        pickle.dump(model_performance, output_file)

    models_counter = 0

    # Inverse modelling
    for n_iterations in n_iterations_list:
        # for batch_size in batch_size_list:  # poki co nie korzystam z batch size!!!
        for n_steps in n_steps_list:
            for n_neurons in n_neurons_list:
                title = "_FRAMED_INVERSE_n_iterations_" + str(n_iterations) + " n_steps_" + str(
                    n_steps) + " n_neurons_" + str(n_neurons)   # + " batch_size_" + str(batch_size) # poki co nie
                                                                # korzystam z batch size
                title = title_addon + title

                init, training_op, X, y, outputs, loss = construct_rnn(n_steps, n_inputs, n_outputs, n_neurons)

                saver = tf.train.Saver()

                mse_train_set, mse_test_set = run_and_plot_rnn_inverse(init, training_op, X, y, outputs, loss, saver, n_iterations, n_steps,
                                                                       control_full, control_full_test, title, full_experiment_length,
                                                                       output_full=output_full, output_full_test=output_full_test,
                                                                       option=option, ploting=ploting,
                                                                       training_signal_addon=training_signal_addon)
                model_inverse_performance.append(
                    {
                        'mse_train_set': mse_train_set,
                        'mse_test_set': mse_test_set,
                        'n_steps': n_steps, 'n_neurons': n_neurons, 'n_iterations': n_iterations,
                        'title': title,
                        'identification_option': option, 'training_signal_addon': training_signal_addon
                    })

                with open(
                        r"model_and_inv_model_identification_mses/model_inverse_performance" + title_addon +
                        training_signal_addon + option + ".pickle",
                        "wb") as output_file:
                    pickle.dump(model_inverse_performance, output_file)

                print("Inverse model " + str(models_counter) + " of " + str(all_models_inverse_counter))
                models_counter = models_counter + 1


if __name__ == "__main__":
    full_experiment_length = 160
    control_full = [1] * int(full_experiment_length)
    control_full_test = control_full[::-1]
    perform_identification(control_full, full_experiment_length, control_full_test, ploting=True)