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
        for i in range(0, experiment_length): #było też n_steps + 1 ale to dla błędnej koncepcji :p
            current_output = simulate_step(control_full[i], previous_output=previous_output)
            y.append(current_output)
            previous_output = current_output
    else:
        y = output_full

    ys = np.asarray([np.asarray(y)])

    control_X_batch = np.asarray([control_full]).reshape(-1, n_steps, 1)
    output_y_batch = ys.reshape(-1, n_steps, 1)

    control_X_batch_flat = control_full
    output_y_batch_flat = y

    return control_X_batch, output_y_batch, previous_output, control_X_batch_flat, output_y_batch_flat # ys1, ys2, previous_output


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
                     control_full, control_full_test, title, experiment_length, output_full=[]):
    with tf.Session() as sess:
        init.run()
        previous_output = 0
        X_batch_full, y_batch_full, previous_output, control_X_batch_flat, output_y_batch_flat = next_batch(
            experiment_length, control_full, n_steps, previous_output, output_full=output_full)  # bylo jeszcze batch_size ale z tego nie korzystam
        X_batch = X_batch_full
        y_batch = y_batch_full

        control_X_batch_flat = [0] * (n_steps - 1) + control_X_batch_flat
        output_y_batch_flat = [0] * (n_steps - 1) + output_y_batch_flat

        for iteration in range(n_iterations):
            print("Normal model " + "./active_suspension_modelling_checkpoints/my_time_series_model"
                  + title + " train, iteration:" + str(iteration))
            for i in range(0, len(control_full) - n_steps):
                temp_y = np.asarray([np.asarray(output_y_batch_flat[i:i + n_steps])]).reshape(-1, n_steps, 1)
                temp_x = np.asarray([np.asarray(control_X_batch_flat[i:i + n_steps])]).reshape(-1, n_steps, 1)
                sess.run(training_op, feed_dict={X: temp_x, y: temp_y})

        saver.save(sess, "./active_suspension_modelling_checkpoints/my_time_series_model" + title)  # not shown in the book
        print("Normal model " + "./active_suspension_modelling_checkpoints/my_time_series_model"
              + title + " done and saved")

    # output_prediction = []
    # ###Testowanie na zbiorze już widzianym (w zasadzie na zbiorze uczącym)
    # with tf.Session() as sess:  # not shown in the book
    #     saver.restore(sess, "./inertia_modelling_checkpoints/my_time_series_model" + title)  # not shown
    #     for i in range(0, len(control_full) - n_steps):
    #         new_temp_x = np.asarray([np.asarray(control_X_batch_flat[i:i + n_steps])]).reshape(-1, n_steps, 1)
    #         y_pred = sess.run(outputs, feed_dict={X: new_temp_x})
    #         output_prediction.append(y_pred[-1][-1])
    #
    # plt.title(title, fontsize=14)
    # plt.plot(range(0, len(output_y_batch_flat) - n_steps), output_y_batch_flat[n_steps:len(output_y_batch_flat)], "b.", markersize=5, label="instance")
    # plt.plot(range(0, len(output_prediction)), output_prediction, "r.", markersize=5, label="prediction")
    # plt.xlabel("Time")
    # plt.legend()
    # plt.show()

    output_prediction_test = []
    previous_output = 0

    ###Testowanie na zbiorze nowym
    with tf.Session() as sess:  # not shown in the book
        saver.restore(sess, "./active_suspension_modelling_checkpoints/my_time_series_model" + title)  # not shown
        for i in range(0, len(control_full) - n_steps):
            new_temp_x = np.asarray([np.asarray(control_full_test[i:i + n_steps])]).reshape(-1, n_steps, 1)
            y_pred_test = sess.run(outputs, feed_dict={X: new_temp_x})
            output_prediction_test.append(y_pred_test[-1][-1])

    unused1, y_batch_test, unused2, control_X_batch_flat, output_y_batch_flat = next_batch(experiment_length, control_full_test, n_steps,
                                                previous_output)
    # plt.title(title, fontsize=14)
    # plt.plot(range(0, len(output_y_batch_flat) - n_steps), output_y_batch_flat[n_steps:], "g.", markersize=5, label="instance_test")
    # plt.plot(range(0, len(output_prediction_test)), output_prediction_test, "m.", markersize=5, label="prediction_test")
    # plt.xlabel("Time")
    # plt.legend()
    # plt.show()

    ##MSE
    mse = math.sqrt(mean_squared_error(output_y_batch_flat[n_steps:], output_prediction_test))

    print("Model " + "./active_suspension_modelling_checkpoints/my_time_series_model"
          + title + " tested. MSE calculated")

    # save_fig(title)
    return mse


def run_and_plot_rnn_inverse(init, training_op, X, y, outputs, loss, saver, n_iterations, n_steps,
                             control_full, control_full_test, title, experiment_length):
    with tf.Session() as sess:
        init.run()
        previous_output = 0
        X_batch_full, y_batch_full, previous_output, control_X_batch_flat, output_y_batch_flat = next_batch(
            experiment_length, control_full, n_steps, previous_output)  # bylo jeszcze batch_size ale z tego nie korzystam
        X_batch = X_batch_full
        y_batch = y_batch_full

        control_X_batch_flat = [0] * (n_steps - 1) + control_X_batch_flat
        output_y_batch_flat = [0] * (n_steps - 1) + output_y_batch_flat

        for iteration in range(n_iterations):
            print("Normal model " + "./active_suspension_modelling_checkpoints/my_time_series_model"
                  + title + " train, iteration:" + str(iteration))
            for i in range(0, len(control_full) - n_steps):
                temp_y = np.asarray([np.asarray(output_y_batch_flat[i:i + n_steps])]).reshape(-1, n_steps, 1)
                temp_x = np.asarray([np.asarray(control_X_batch_flat[i:i + n_steps])]).reshape(-1, n_steps, 1)
                sess.run(training_op, feed_dict={X: temp_y, y: temp_x})

        saver.save(sess, "./active_suspension_modelling_checkpoints/my_time_series_model" + title)  # not shown in the book

    # input_prediction = []
    # ###Testowanie na zbiorze już widzianym (w zasadzie na zbiorze uczącym)
    # with tf.Session() as sess:  # not shown in the book
    #     saver.restore(sess, "./inertia_modelling_checkpoints/my_time_series_model" + title)  # not shown
    #     for i in range(0, len(control_full) - n_steps):
    #         new_temp_y = np.asarray([np.asarray(output_y_batch_flat[i:i + n_steps])]).reshape(-1, n_steps, 1)
    #         X_pred = sess.run(outputs, feed_dict={X: new_temp_y})
    #         input_prediction.append(X_pred[-1][-1])
    #
    # plt.title(title, fontsize=14)
    # plt.plot(range(0, len(control_X_batch_flat) - n_steps), control_X_batch_flat[n_steps:len(control_X_batch_flat)], "b.", markersize=10, label="instance")
    # plt.plot(range(0, len(input_prediction)), input_prediction, "r.", markersize=10, label="prediction")
    # plt.xlabel("Time")
    # plt.show()

    input_prediction_test = []
    previous_output = 0

    X_batch_full, y_batch_full, previous_output, control_X_batch_flat, output_y_batch_flat_test = next_batch(
        experiment_length, control_full_test, n_steps, previous_output)  # bylo jeszcze batch_size ale z tego nie korzystam

    ###Testowanie na zbiorze nowym
    with tf.Session() as sess:  # not shown in the book
        saver.restore(sess, "./active_suspension_modelling_checkpoints/my_time_series_model" + title)  # not shown
        for i in range(0, len(control_full) - n_steps):
            new_temp_y = np.asarray([np.asarray(output_y_batch_flat_test[i:i + n_steps])]).reshape(-1, n_steps, 1)
            X_pred_test = sess.run(outputs, feed_dict={X: new_temp_y})
            input_prediction_test.append(X_pred_test[-1][-1])
            print("Inverse model " + "./active_suspension_modelling_checkpoints/my_time_series_model"
            + title + " test, iteration:" + str(i))

    # plt.title()
    # plt.plot(range(0, len(control_full_test) - n_steps), control_full_test[n_steps:], "g.", markersize=10, label="instance_test")
    # plt.plot(range(0, len(input_prediction_test)), input_prediction_test, "m.", markersize=10, label="prediction_test")
    # plt.xlabel("Time")
    # plt.show()

    ##MSE
    mse = math.sqrt(mean_squared_error(control_full_test[n_steps:], input_prediction_test))

    # save_fig(title)
    return mse


def perform_identification(control_full, full_experiment_length, control_full_test, title_addon='',
                           n_iterations_list=[10, 20, 50],  # czyli ile razy przejeżdżam przez cały zbiór danych
                           # batch_size_list=[2, 25],  # , 100]
                           n_neurons_list=[50, 200, 500],  # [1, 10, 100]
                           n_steps_list=[10, 20, 30],  # czyli ile poprzednich sterowań biorę pod uwagę
                           mode='basicRNN',
                           option='', output_full=[]):
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

                saver = tf.train.Saver()

                mse = run_and_plot_rnn(init, training_op, X, y, outputs, loss, saver, n_iterations, n_steps,
                                 control_full, control_full_test, title, full_experiment_length, output_full=output_full)
                model_performance.append(
                    {
                        'MSE': mse,
                        'n_steps': n_steps, 'n_neurons': n_neurons, 'n_iterations': n_iterations,
                        'identification_option': option
                    })
                print("Model " + str(models_counter) + " of " + str(all_models_counter))
                models_counter = models_counter + 1

    models_counter = 0

    ##Inverse modelling
    for n_iterations in n_iterations_list:
    #    for batch_size in batch_size_list:  # poki co nie korzystam z batch size!!!
        for n_steps in n_steps_list:
            for n_neurons in n_neurons_list:
                title = "_FRAMED_INVERSE_n_iterations_" + str(n_iterations) + " n_steps_" + str(
                    n_steps) + " n_neurons_" + str(n_neurons)  #  + " batch_size_" + str(batch_size) # poki co nie korzystam z batch size
                title = title_addon + title

                init, training_op, X, y, outputs, loss = construct_rnn(n_steps, n_inputs, n_outputs, n_neurons)

                saver = tf.train.Saver()

                mse = run_and_plot_rnn_inverse(init, training_op, X, y, outputs, loss, saver, n_iterations, n_steps,
                                         control_full, control_full_test, title, full_experiment_length)
                model_inverse_performance.append(
                    {
                        'MSE': mse,
                        'n_steps': n_steps, 'n_neurons': n_neurons, 'n_iterations': n_iterations,
                        'identification_option': option
                    })
                print("Inverse model " + str(models_counter) + " of " + str(all_models_inverse_counter))
                models_counter = models_counter + 1

    with open(r"model_performance.pickle", "wb") as output_file:
        pickle.dump(model_performance, output_file)

    with open(r"model_inverse_performance.pickle", "wb") as output_file:
        pickle.dump(model_inverse_performance, output_file)


if __name__ == "__main__":
    full_experiment_length = 160
    control_full = [1] * int(full_experiment_length)
    control_full_test = control_full[::-1]
    perform_identification(control_full, full_experiment_length, control_full_test)