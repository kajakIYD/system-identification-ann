import tensorflow as tf

import numpy as np
import matplotlib.pyplot as plt
import pickle
from pathlib import Path
from sklearn.metrics import mean_squared_error

import inertia_modelling
import model_and_inv_model_identification

import subprocess
import os
import signal

import socket
HOST = '127.0.0.1'  # Standard loopback interface address (localhost)
PORT = 8123        # Port to listen on (non-privileged ports are > 1023)


def extract_rnn_structure_from_title(title):
    split_title = title.split(" ")

    for chunk in split_title:
        if "n_neurons" in chunk:
            n_neurons = int(chunk[chunk.index("n_neurons_") + len("n_neurons_"):])

        if "n_steps" in chunk:
            n_steps = int(chunk[chunk.index("n_steps_") + len("n_steps_"):])

        if "n_iterations" in chunk:
            n_iterations = int(chunk[chunk.index("n_iterations_") + len("n_iterations_"):])

    return n_neurons, n_steps, n_iterations



def extract_models_and_inverse_models_data(directory_in_str="./inertia_modelling_checkpoints/"):
    titles_model_inverse_data = []
    titles_model_data = []

    pathlist = Path(directory_in_str).glob('**/*.meta')
    for path in pathlist:
        # because path is object not string
        path_in_str = str(path).split(".")[0]
        title = path_in_str

        n_neurons, n_steps, n_iterations = extract_rnn_structure_from_title(title)

        if "INVERSE" in path_in_str:
            titles_model_inverse_data.append({'title': title, 'n_neurons': n_neurons,
                                      'n_steps': n_steps, 'n_iterations': n_iterations})
        else:
            titles_model_data.append({'title': title, 'n_neurons': n_neurons,
                                      'n_steps': n_steps, 'n_iterations': n_iterations})

    return titles_model_inverse_data, titles_model_data


def unpickle_model_and_model_inverse_performance():
    with open(r"model_performance.pickle", "rb") as input_file:
        model_performance = pickle.load(input_file)

    with open(r"model_inverse_performance.pickle", "rb") as input_file:
        model_inverse_performance = pickle.load(input_file)

    return model_inverse_performance, model_performance


def pickle_object(object, file_name="pickled_object.pkl"):
    with open(file_name, "wb") as output_file:
        pickle.dump(object, output_file)


sim_time_const = 30
sample_rate_hz_const = 500
def main(titles_model_inverse_data, titles_model_data, simulation_time=sim_time_const,
         dt=0.1, SP=sim_time_const * [0], mse_calc=True,
         plotting=False, suspension_simulation=False):
    # SP = model_and_inv_model_identification.generate_sine(int(30 / 0.1), 1, 0.1)
    # model_inverse_performance, model_performance = unpickle_model_and_model_inverse_performance()

    plt.rcParams.update({'font.size': 6})

    mses = []

    models_loop_counter = 0

    for title_model_inverse_data in titles_model_inverse_data:
        for title_model_data in titles_model_data:
            n_steps_inverse_model = title_model_inverse_data['n_steps']
            n_steps_model = title_model_data['n_steps']

            n_inputs = 1
            n_outputs = 1

            n_iterations_model_inverse = title_model_inverse_data['n_iterations']
            n_iterations_model = title_model_data['n_iterations']

            n_neurons_inverse_model = title_model_inverse_data['n_neurons']
            n_neurons_model = title_model_data['n_neurons']

            title_model_inverse = title_model_inverse_data['title']
            title_model = title_model_data['title']

            init_model_inverse, training_op, X_model_inverse, y, outputs_inverse, loss = inertia_modelling.construct_rnn(n_steps_inverse_model, n_inputs,
                                                                                             n_outputs, n_neurons_inverse_model)

            saver = tf.train.Saver()

            sess_model_inverse = tf.Session()
            init_model_inverse.run(session=sess_model_inverse)
            saver.restore(sess_model_inverse,
                          title_model_inverse)

            init_model, training_op, X_model, y, outputs, loss = inertia_modelling.construct_rnn(n_steps_model, n_inputs,
                                                                                     n_outputs, n_neurons_model)

            saver = tf.train.Saver()

            sess_model = tf.Session()
            init_model.run(session=sess_model)
            saver.restore(sess_model,
                          title_model)

            previous_model_plant_disturbed_difference = [0] * n_steps_inverse_model
            inverse_model_input_vector = [0] * n_steps_inverse_model
            model_input_vector = [0] * n_steps_model

            plant_output = [0]
            plant_control = []
            disturbed_plant_output = []
            model_plant_disturbed_difference = []
            SP_feedback_difference = []

            loop_counter = 1

            # inertia parameters
            a = 1
            b = 0.1

            if suspension_simulation:
                path_to_simulator_executable = "/home/user/Documents/simEnv_2018_07_31/simProgram"
                sample_rate_hz = sample_rate_hz_const
                length_of_experiment = simulation_time * 400
                meas_dest_file_name = "/home/user/Documents/simEnv_2018_07_31/simResults/" \
                                      + title_model.split('/')[1].replace(' ', '_') \
                                      + title_model_inverse.split('/')[1].replace(' ', '_')
                mr_control_parameters_file_name = "/home/user/Documents/simEnv_2018_07_31/ctrl_params_tmp"
                args_list = [path_to_simulator_executable, str(sample_rate_hz), str(length_of_experiment),
                             meas_dest_file_name, mr_control_parameters_file_name]
                subprocess.Popen(args_list)  # run process in background
                s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                s.bind((HOST, PORT))
                s.listen()
                conn, addr = s.accept()
                print("s.accept performed!")

            for t in range(0, int(simulation_time / dt)):
                inverse_model_input_vector[-1] = SP[t] - previous_model_plant_disturbed_difference[-1]
                model_plant_disturbed_difference.append(previous_model_plant_disturbed_difference[-1])
                SP_feedback_difference.append(inverse_model_input_vector[-1])

                X_new_model_inverse = np.asarray(inverse_model_input_vector).reshape(-1, n_steps_inverse_model, 1)
                y_pred_inverse_model = sess_model_inverse.run(outputs_inverse, feed_dict={X_model_inverse: X_new_model_inverse})

                model_input_vector = model_input_vector[1:]
                model_input_vector.append(y_pred_inverse_model[-1][-1])

                X_new_model = np.asarray(model_input_vector).reshape(-1, n_steps_model, 1)
                y_pred_model = sess_model.run(outputs, feed_dict={X_model: X_new_model})

                current_control = y_pred_inverse_model[-1][-1]
                plant_control.append(current_control)
                if suspension_simulation:
                    # send control
                    conn.sendall(bytearray(str(current_control), 'utf-8'))

                    #receive suspension output
                    plant_output.append(float(conn.recv(1024)))
                else:
                    plant_output.append(inertia_modelling.simulate_step(current_control,
                                                                        plant_output[loop_counter - 1], a, b))

                disturbances = 0
                disturbed_plant_output.append(plant_output[loop_counter] + disturbances)

                previous_model_plant_disturbed_difference = previous_model_plant_disturbed_difference[1:]
                previous_model_plant_disturbed_difference.append(disturbed_plant_output[loop_counter - 1] - y_pred_model[-1][-1])

                inverse_model_input_vector = inverse_model_input_vector[1:]
                inverse_model_input_vector.append(0)

                loop_counter = loop_counter + 1

                if loop_counter % 1000 == 0:
                    print(str(loop_counter))

            if suspension_simulation:
                conn.sendall(b'$')
                print("C suspension simulator killed!")

            loop_counter = loop_counter - 1

            if plotting == True:
                plt.plot(range(0, loop_counter), disturbed_plant_output, "b.", label="Disturbed plant output")
                plt.plot(range(0, loop_counter), plant_control, "r.", label="Inverse model output (control)")
                plt.plot(range(0, loop_counter), model_plant_disturbed_difference, "g.", label="Disturbed plant - model_output")
                plt.plot(range(0, loop_counter), SP_feedback_difference, "m.", label="SP_feedback_difference")
                plt.plot(range(0, loop_counter), SP, "y-", label="SP")
                plt.legend()
                plt.xlabel("Time")
                plt.title("model_inverse: neurons" + str(n_neurons_inverse_model) + " steps" + str(n_steps_inverse_model)
                          + " n_iterations:" + str(n_iterations_model_inverse) + "model: neurons "
                          + str(n_neurons_model) + " steps" + str(n_steps_inverse_model)
                          + " n_iterations:" + str(n_iterations_model))
                plt.show()

            models_loop_counter = models_loop_counter + 1

            print("Model: " + str(models_loop_counter) + " out of: " +
                  str(len(titles_model_data) * len(titles_model_inverse_data)))

            if mse_calc == True:
                if not (True in np.isnan(disturbed_plant_output)):
                    mses.append({'mse': mean_squared_error(disturbed_plant_output, SP),
                                'model_title': title_model, 'model_inverse_title': title_model_inverse})

                if (models_loop_counter % 50 and not models_loop_counter == 0) \
                        or (len(titles_model_data) * len(titles_model_inverse_data) - models_loop_counter) < 50:
                    pickle_object(mses, "mses_" + str(models_loop_counter) + ".pkl")
                    mses.clear()

    if mse_calc == True:
        mses_vals = [item['mse'] for item in mses]

        min_mse_index = mses_vals.index(min(mses_vals))

        print(mses[min_mse_index])

        pickle_object(mses, "mses.pkl")



if __name__ == "__main__":
    titles_model_inverse_data, titles_model_data = extract_models_and_inverse_models_data()
    main(titles_model_inverse_data, titles_model_data)