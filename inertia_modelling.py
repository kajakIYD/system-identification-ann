import tensorflow as tf
from tensorflow.keras import models
from tensorflow.keras import layers
import keras
from sklearn.metrics import mean_squared_error

import math
import numpy as np
import matplotlib.pyplot as plt
import os
import pickle

import model_and_inv_model_identification


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


def simulate_inertia_for_given_control(control_full, previous_output=0):
    y = []
    for i in range(0, len(control_full)):  # było też n_steps + 1 ale to dla błędnej koncepcji :p
        current_output = simulate_step(control_full[i], previous_output=previous_output)
        y.append(current_output)
        previous_output = current_output

    return y


def next_batch(control_full, n_steps, previous_output=0):
    if not len(control_full) % n_steps == 0:
        modulo = len(control_full) % n_steps
        control_full = control_full[:-modulo]

    y = simulate_inertia_for_given_control(control_full)

    ys = np.asarray([np.asarray(y)])

    control_X_batch = np.asarray([control_full]).reshape(-1, n_steps, 1)
    output_y_batch = ys.reshape(-1, n_steps, 1)

    return control_X_batch, output_y_batch, previous_output# ys1, ys2, previous_output


def construct_rnn(n_steps, n_inputs, n_outputs, n_neurons):
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


def evaluate_model(saver, control_full, path_to_checkpoints, title, n_steps, outputs, X):
    if not len(control_full) % n_steps == 0:
        modulo = len(control_full) % n_steps
        control_full = control_full[:-modulo]

    model_input_vector = [0] * n_steps + control_full
    model_output_vector = []


    with tf.Session() as sess:
        saver.restore(sess, path_to_checkpoints + "/my_time_series_model" + title)  # not shown
        for control in control_full:
            X_new_model = np.asarray(model_input_vector).reshape(-1, n_steps, 1)
            y_pred_model = sess.run(outputs, feed_dict={X: X_new_model})
            model_output_vector.append(y_pred_model[-1][-1])
            model_input_vector = model_input_vector[1:]
            model_input_vector.append(control)

    return model_output_vector


def evaluate_model_inverse(saver, control_full, path_to_checkpoints, title, n_steps, outputs, X):
    if not len(control_full) % n_steps == 0:
        modulo = len(control_full) % n_steps
        control_full = control_full[:-modulo]

    model_inverse_input_vector = simulate_inertia_for_given_control(control_full)

    object_output_full = model_inverse_input_vector
    model_inverse_output_vector = []

    with tf.Session() as sess:
        saver.restore(sess, path_to_checkpoints + "/my_time_series_model" + title)  # not shown
        for object_output in object_output_full:
            X_new_model = np.asarray(model_inverse_input_vector).reshape(-1, n_steps, 1)
            y_pred_model = sess.run(outputs, feed_dict={X: X_new_model})
            model_inverse_output_vector.append(y_pred_model[-1][-1])
            model_inverse_input_vector = model_inverse_input_vector[1:]
            model_inverse_input_vector.append(object_output)

    return model_inverse_output_vector


def run_and_plot_rnn(init, training_op, X, y, outputs, loss, saver, n_iterations, n_steps,
                    control_full, control_full_test, title,
                    output_full = [], output_full_test = [],
                    option='inertia_modelling', ploting=False, training_signal_addon = '',
                    path_to_checkpoints="./inertia_modelling_checkpoints_trying_to_reproduce"):

    with tf.Session() as sess:
        init.run()
        previous_output = 0
        X_batch_full, y_batch_full, previous_output = next_batch(control_full, n_steps,
                                                                 previous_output)
        X_batch = X_batch_full
        y_batch = y_batch_full
        for iteration in range(n_iterations):
            sess.run(training_op, feed_dict={X: X_batch, y: y_batch})
            # if iteration % 100 == 0:
            #     mse = loss.eval(feed_dict={X: X_batch, y: y_batch})
            #     print(iteration, "\tMSE:", mse)

        saver.save(sess, path_to_checkpoints + "/my_time_series_model" + title)  # not shown in the book

    # Testowanie na zbiorze już widzianym (w zasadzie na zbiorze uczącym)
    model_output_vector = evaluate_model(saver, control_full, path_to_checkpoints, title, n_steps, outputs, X)

    y_batch_flat = [item for sub_y_batch in y_batch for item in sub_y_batch]
    mse_training_set = mean_squared_error(y_batch_flat, model_output_vector)

    if ploting:
        plt.title(title, fontsize=14)
        plt.plot(y_batch_flat, "b-", markersize=10, label="instance")
        plt.plot(model_output_vector, "r-", markersize=10, label="prediction")
        plt.legend()
        plt.grid()
        plt.show()

    control_full_test = model_and_inv_model_identification.generate_test_signal()
    previous_output = 0

    # Testowanie na zbiorze nowym
    model_output_vector = evaluate_model(saver, control_full_test, path_to_checkpoints, title, n_steps, outputs, X)

    unused1, y_batch_test, unused2 = next_batch(control_full_test, n_steps,
                                                previous_output)
    y_batch_test_flat = [item for sub_y_batch in y_batch_test for item in sub_y_batch]

    mse_test_set = mean_squared_error(y_batch_test_flat, model_output_vector)

    if ploting:
        plt.plot(y_batch_test_flat, "b-", markersize=10, label="instance_test")
        plt.plot(model_output_vector, "r-", markersize=10, label="prediction_test")
        plt.legend()
        plt.grid()
        plt.xlabel("Time")
        # save_fig(title)
        plt.show()

    return mse_training_set, mse_test_set


def run_and_plot_rnn_inverse(init, training_op, X, y, outputs, loss, saver, n_iterations, n_steps,
                             control_full, control_full_test, title,
                             output_full=[], output_full_test=[],
                             option='inertia_modelling', ploting=False,
                             training_signal_addon='',
                             path_to_checkpoints="./inertia_modelling_checkpoints_trying_to_reproduce"):
    with tf.Session() as sess:
        init.run()
        previous_output = 0
        X_batch_full, y_batch_full, previous_output = next_batch(control_full, n_steps,
                                                                 previous_output)
        for iteration in range(n_iterations):
            sess.run(training_op, feed_dict={X: y_batch_full, y: X_batch_full})
            # if iteration % 100 == 0:
            #     mse = loss.eval(feed_dict={X: X_batch, y: y_batch})
            #     print(iteration, "\tMSE:", mse)

        saver.save(sess, path_to_checkpoints + "/my_time_series_model" + title)  # not shown in the book

    # Testowanie na zbiorze już widzianym (w zasadzie na zbiorze uczącym)
    model_inverse_output_vector = evaluate_model_inverse(saver, control_full, path_to_checkpoints, title, n_steps,
                                                         outputs, X)

    mse_training_set = mean_squared_error(model_inverse_output_vector, control_full)

    if ploting:
        plt.title(title, fontsize=14)
        plt.plot(control_full, "b-", markersize=10, label="instance")
        plt.plot(model_inverse_output_vector, "r-", markersize=10, label="prediction")
        plt.show()

    # Testowanie na zbiorze nowym
    model_inverse_output_vector = evaluate_model_inverse(saver, control_full_test, path_to_checkpoints, title, n_steps,
                                                         outputs, X)

    modulo = len(control_full_test) % n_steps
    if not modulo == 0:
        mse_test_set = mean_squared_error(control_full_test[:-modulo], model_inverse_output_vector)
    else:
        mse_test_set = mean_squared_error(control_full_test, model_inverse_output_vector)

    if ploting:
        plt.plot(control_full_test[:-modulo], "b-", markersize=10, label="instance_test")
        plt.plot(model_inverse_output_vector, "r-", markersize=10, label="prediction_test")
        plt.xlabel("Time")
        # save_fig(title)
        plt.show()

    return mse_training_set, mse_test_set


def perform_identification(control_full, experiment_length, control_full_test, output_full=[], ploting=False,
                           n_iterations_list=[10, 20, 50],
                           n_neurons_list=[50, 200, 500, 1000],  # [1, 10, 100]
                           n_steps_list=[10, 20, 50],
                           training_signal_addon="_trying_to_reproduce_MIXED_TRAINED",
                           title_addon="_trying_to_reproduce_MIXED_TRAINED",
                           path_to_checkpoints="./inertia_modelling_checkpoints_trying_to_"
                                               "reproduce_MIXED_TRAINED"):
    n_inputs = 1
    n_outputs = 1

    all_models_counter = len(n_iterations_list) * len(n_steps_list) * len(n_neurons_list)
    models_counter = 1
    model_performance = []



    for n_iterations in n_iterations_list:
        for n_steps in n_steps_list:
            for n_neurons in n_neurons_list:

                title = "n_iterations_" + str(n_iterations) + " n_steps_" + str(
                    n_steps) + " n_neurons_" + str(n_neurons)

                init, training_op, X, y, outputs, loss = construct_rnn(n_steps, n_inputs, n_outputs, n_neurons)

                saver = tf.train.Saver()

                print(title)
                mse_training_set, mse_test_set = run_and_plot_rnn(init, training_op, X, y, outputs, loss, saver,
                                                                  n_iterations, n_steps, control_full, control_full_test,
                                                                  title, ploting=ploting,
                                                                  path_to_checkpoints=path_to_checkpoints)

                model_performance.append(
                    {
                        'mse_test_set': mse_test_set, 'mse_training_set': mse_training_set,
                        'n_steps': n_steps, 'n_neurons': n_neurons, 'n_iterations': n_iterations,
                        'title': title,
                        'identification_option': "", 'training_signal_addon': training_signal_addon
                    })
                print("Model " + str(models_counter) + " of " + str(all_models_counter) + " DONE")
                models_counter = models_counter + 1

                with open(r"model_and_inv_model_identification_mses/model_performance" + title_addon + ".pickle",
                          "wb") as output_file:
                    pickle.dump(model_performance, output_file)

    models_counter = 1

    model_inverse_performance = []
    # Inverse modelling
    for n_iterations in n_iterations_list:
        for n_steps in n_steps_list:
            for n_neurons in n_neurons_list:
                title = "INVERSE_n_iterations_" + str(n_iterations) + " n_steps_" + str(
                    n_steps) + " n_neurons_" + str(n_neurons)  #  + " batch_size_" + str(batch_size) # poki co nie korzystam z batch size
                # plot_training_instance()

                init, training_op, X, y, outputs, loss = construct_rnn(n_steps, n_inputs, n_outputs, n_neurons)

                saver = tf.train.Saver()

                mse_training_set, mse_test_set = run_and_plot_rnn_inverse(init, training_op, X, y, outputs, loss, saver,
                                                                          n_iterations, n_steps, control_full,
                                                                          control_full_test, title, ploting=ploting,
                                                                          path_to_checkpoints=path_to_checkpoints)
                print(title)
                model_inverse_performance.append(
                    {
                        'mse_training_set': mse_training_set, 'mse_test_set': mse_test_set,
                        'n_steps': n_steps, 'n_neurons': n_neurons, 'n_iterations': n_iterations,
                        'title': title,
                        'identification_option': "", 'training_signal_addon': training_signal_addon
                    })
                print("Model inverse " + str(models_counter) + " of " + str(all_models_counter) + " DONE")
                models_counter = models_counter + 1

                with open(r"model_and_inv_model_identification_mses/model_inverse_performance" + title_addon + ".pickle",
                          "wb") as output_file:
                    pickle.dump(model_inverse_performance, output_file)


if __name__ == "__main__":
    control_full = model_and_inv_model_identification.generate_identification_signal_1()
    control_full_test = model_and_inv_model_identification.generate_test_signal()
    output_full = simulate_inertia_for_given_control(control_full)
    perform_identification(control_full, control_full_test, output_full, ploting=False)
