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


def calc_ise(output, SP, dt):
    ise = 0
    for out_probe, SP_probe in zip(output, SP):
        error = (out_probe - SP_probe) ** 2 * dt
        ise = ise + error

    return ise


def calc_iae(output, SP, dt):
    iae = 0
    for out_probe, SP_probe in zip(output, SP):
        error = abs(out_probe - SP_probe) * dt
        iae = iae + error

    return iae


def calc_itae(output, SP, dt):
    itae = 0
    probes_count = 1
    for out_probe, SP_probe in zip(output, SP):
        error = abs(out_probe - SP_probe) * dt
        itae = itae + probes_count * dt * error

    return itae


sim_time_const = 10
sample_rate_hz_const = 500
def main(titles_model_inverse_data, titles_model_data, simulation_time=sim_time_const,
         dt=0.1, SP=sim_time_const * [0], mse_calc=True,
         plotting=False, suspension_simulation=False, path_to_save_mses=''):
    # SP = model_and_inv_model_identification.generate_sine(int(30 / 0.1), 1, 0.1)
    # model_inverse_performance, model_performance = unpickle_model_and_model_inverse_performance()

    plt.rcParams.update({'font.size': 6})

    mses = []
    mses_all = []

    models_loop_counter = 0
    all_models_counter = len(titles_model_data) * len(titles_model_inverse_data)

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
            if not suspension_simulation:
                a = 1
                b = 0.1

            if suspension_simulation:
                path_to_simulator_executable = "/home/user/Documents/simEnv_2018_07_31/simProgram"
                sample_rate_hz = sample_rate_hz_const
                meas_dest_file_name = "/home/user/Documents/simEnv_2018_07_31/simResults/" \
                                      + title_model.split('/')[1].replace(' ', '_') \
                                      + title_model_inverse.split('/')[1].replace(' ', '_')
                mr_control_parameters_file_name = "/home/user/Documents/simEnv_2018_07_31/ctrl_params_tmp"
                args_list = [path_to_simulator_executable, str(sample_rate_hz), str(simulation_time),
                             meas_dest_file_name, mr_control_parameters_file_name]
                subprocess.Popen(args_list)  # run process in background
                s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                s.bind((HOST, PORT))
                s.listen()
                print("Python Simulation Server awaiting for connection...")
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

                current_control = float(y_pred_inverse_model[-1][-1])
                plant_control.append(current_control)
                if suspension_simulation:
                    # send control
                    if conn.sendall(str(current_control).encode()) is None:
                        # conn.sendall(bytearray(str(1111.11), 'utf-8'))
                        try:
                            #receive suspension output
                            received = conn.recv(1024)
                            control_float = float(received)
                        except ConnectionResetError:
                            print("Stopped at probe " + str(t))
                        except ValueError:
                            print("PROBLEM@!!! OTRZYMANO: " + received)
                            return
                        #print(received)
                        plant_output.append(control_float)
                else:
                    plant_output.append(inertia_modelling.simulate_step(current_control,
                                                                        plant_output[loop_counter - 1], a, b))

                disturbances = 0
                disturbed_plant_output.append(plant_output[-1] + disturbances)

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

            if plotting:
                fig, ax1 = plt.subplots()
                ax1.plot(range(0, len(disturbed_plant_output)), disturbed_plant_output, "b.", label="Disturbed plant output")
                ax1.plot(range(0, len(SP)), SP, "y-", label="SP")
                ax1.plot(range(0, len(model_plant_disturbed_difference)), model_plant_disturbed_difference, "g.", label="Disturbed plant - model_output")
                ax1.plot(range(0, len(SP_feedback_difference)), SP_feedback_difference, "m.", label="SP_feedback_difference")
                ax1.set_xlabel('Probes')
                # Make the y-axis label, ticks and tick labels match the line color.
                ax1.set_ylabel('velocity, m/s', color='b')
                ax1.tick_params('y', colors='b')
                ax2 = ax1.twinx()
                ax2.plot(range(0, len(plant_control)), plant_control, "r.", label="Inverse model output (control)")
                ax2.set_ylabel('Control', color='r')
                ax2.tick_params('y', colors='r')
                fig.tight_layout()
                plt.legend()
                plt.xlabel("Time")
                plt.title("model_inverse: neurons" + str(n_neurons_inverse_model) + " steps" + str(n_steps_inverse_model)
                          + " n_iterations:" + str(n_iterations_model_inverse) + "model: neurons "
                          + str(n_neurons_model) + " steps" + str(n_steps_inverse_model)
                          + " n_iterations:" + str(n_iterations_model))
                plt.show()

            models_loop_counter = models_loop_counter + 1

            print("Model: " + str(models_loop_counter) + " out of: " +
                  str(all_models_counter))

            if mse_calc:
                if not (True in np.isnan(disturbed_plant_output)):
                    mses.append({'mse': mean_squared_error(disturbed_plant_output, SP),
                                 'ise': calc_ise(disturbed_plant_output, SP, dt),
                                 'iae': calc_iae(disturbed_plant_output, SP, dt),
                                 'itae': calc_itae(disturbed_plant_output, SP, dt),
                                'model_title': title_model, 'model_inverse_title': title_model_inverse})

                if (models_loop_counter % 50 == 0 and not models_loop_counter == 0) \
                        or (all_models_counter - models_loop_counter) < 50:
                    pickle_object(mses, path_to_save_mses + "mses_" + str(models_loop_counter) + ".pkl")
                    mses_all = mses_all + mses
                    mses.clear()

    if mse_calc:
        mses_vals = [item['mse'] for item in mses_all]

        min_mse_index = mses_vals.index(min(mses_vals))

        print(mses_all[min_mse_index])

        pickle_object(mses, "mses.pkl")


def compile_proper_simulator_in_TCP_mode(mode='active_suspension'):
    # mode='semi_active_suspension'

    # make change in configuration.h (enable TCP mode and proper suspension model)
    fileName = '/home/user/Documents/simEnv_2018_07_31/configuration.h'

    with open(fileName, "r") as file:
        content = file.read()

    content_modified = ''
    for line in content.split('\n'):
        if r"//#define TCP_ONLINE_SIMULATION" in line:
            line = line.replace(r"//#define TCP_ONLINE_SIMULATION", r"#define TCP_ONLINE_SIMULATION")
        if r"#define ROAD_EXC_OFF" in line and r"//#define ROAD_EXC_OFF" not in line:
            line = line.replace(r"#define ROAD_EXC_OFF", r"//#define ROAD_EXC_OFF")
        if r"//#define SIN_FREQ_ROAD_EXC" in line:
            line = line.replace(r"//#define SIN_FREQ_ROAD_EXC", r"#define SIN_FREQ_ROAD_EXC")
        if r"//#define NO_BYPASS_CONTROL" in line:
            line = line.replace(r"//#define NO_BYPASS_CONTROL", r"#define NO_BYPASS_CONTROL")

        content_modified = content_modified + line + '\n'

    with open(fileName, "w") as file:
        file.write(content_modified)

    # make change in Makefile
    fileName = '/home/user/Documents/simEnv_2018_07_31/Makefile'

    with open(fileName, "r") as file:
        content = file.read()

    content_modified = ''
    for line in content.split('\n'):
        if r"ROAD_EXC = roadExcOff" in line and not r"#ROAD_EXC = roadExcOff" in line:
            line = line.replace(r"ROAD_EXC = roadExcOff", r"#ROAD_EXC = roadExcOff")
        if r"#ROAD_EXC = sinFreqRoadExc" in line:
            line = line.replace(r"#ROAD_EXC = sinFreqRoadExc", r"ROAD_EXC = sinFreqRoadExc")
        if r"#ROAD_EXC = tcp_control" in line:
            line = line.replace(r"#ROAD_EXC = tcp_control", r"ROAD_EXC = tcp_control")


        content_modified = content_modified + line + '\n'

    with open(fileName, "w") as file:
        file.write(content_modified)

    # make clean
    args_list = ['make', 'clean', '-C', '/home/user/Documents/simEnv_2018_07_31']
    subprocess.run(args_list)

    # make all
    args_list = ['make', 'all', '-C', '/home/user/Documents/simEnv_2018_07_31']
    subprocess.run(args_list)


def simulation_TCP(title_model_inverse_data, title_model_data, mse_calc=False, plotting=True):
    compile_proper_simulator_in_TCP_mode(mode='active_suspension')

    main(title_model_inverse_data, title_model_data, dt=1 / sample_rate_hz_const, simulation_time=sim_time_const,
         SP=sim_time_const * sample_rate_hz_const * [0], mse_calc=mse_calc, suspension_simulation=True,
         plotting=plotting)


if __name__ == "__main__":
    compile_proper_simulator_in_TCP_mode(mode='active_suspension')

    titles_model_inverse_data, titles_model_data = extract_models_and_inverse_models_data(
                                                   directory_in_str="/home/user/Documents/test")# ./active_suspension_modelling_checkpoints")
    # titles_model_inverse_data, titles_model_data = extract_models_and_inverse_models_data\
    #                                                 ("./inertia_modelling_checkpoints")

    main(titles_model_inverse_data, titles_model_data, dt=1/sample_rate_hz_const, simulation_time=sim_time_const,
         SP=sim_time_const * sample_rate_hz_const * [0], suspension_simulation=True, plotting=False,
         path_to_save_mses='/home/user/Documents/system-identification-ann/active_suspension_simulation_performances/')
