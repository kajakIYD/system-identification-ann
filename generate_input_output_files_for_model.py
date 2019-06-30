# Create.txt file with force that should be applied to damper in each line
from math import sin
import subprocess
from matplotlib import pyplot as plt
import time
import os

import model_and_inv_model_identification
import simulation


def generate_sine_force_chunk(N, Amp, omega, bias, fid, pi=3.1415):
    force_vector = []
    for j in range(1, N + 1):
        value = Amp * sin(omega * j / (2 * pi)) + bias
        fid.write(str(value) + '\n')
        force_vector.append(value)

    return force_vector


def create_force_external_files(freq, sim_time_in_sec, scenarios_cnt=5):
    Amp_max = 100000

    force_external_files = []
    force_vectors_dict = dict()

    N = int(freq * sim_time_in_sec / scenarios_cnt)

    for i in range(1, scenarios_cnt + 1):

        wholeN = N * 5

        fileName = '/home/user/Documents/simEnv_2018_07_31/forceExternal_len_' + str(N) + '_' + str(i) + '.txt'
        force_external_files.append(fileName)

        force_vector = []

        with open(fileName, "w") as fid:

            Amp = Amp_max / 1.1 / i
            omega = 0.1 / i
            bias = 1000 / i

            force_vector = force_vector + generate_sine_force_chunk(N, Amp, omega, bias, fid)

            Amp = Amp_max / 1.2 / i
            omega = 0.01 / i
            bias = 100 / i

            # force_vector = force_vector + generate_sine_force_chunk(N, Amp, omega, bias, fid)

            Amp = Amp_max / 1.3 / i
            omega = 0.05 / i
            bias = 500 / i

            # force_vector = force_vector + generate_sine_force_chunk(N, Amp, omega, bias, fid)

            Amp = Amp_max / 1.4 / i
            omega = 0.02 / i
            bias = 500 / i

            # force_vector = force_vector + generate_sine_force_chunk(N, Amp, omega, bias, fid)

            Amp = Amp_max / 0.5 / i
            omega = 0.03 / i
            bias = 300 / i

            # force_vector = force_vector + generate_sine_force_chunk(N, Amp, omega, bias, fid)

            force_vectors_dict[fileName] = force_vector

    return force_external_files, force_vectors_dict


def write_velocity_to_file(force_file_directory, force_file_name, velocity):
    velocity_file_name = force_file_directory + "/PythonAutomation/" + force_file_name.split('.')[0] + "_vel.txt"

    with open(velocity_file_name, 'w') as file:
        for vel in velocity:
            file.write(str(vel) + '\n')

    return velocity_file_name


def run_simPrograms(force_external_files, freq, sim_time_in_sec):
    velocity_files = []
    sim_results_dict = dict()

    for force_file in force_external_files:
        force_file_directory = ""
        for chunk in force_file.split('/')[1:-1]:
            force_file_directory = force_file_directory + "/" + chunk

        force_file_name_clean = force_file.split('/')[-1]
        meas_dest_file_name = force_file_directory + "/PythonAutomation/" + "meas_" \
                              + force_file.split('/')[-1].split('.')[0]
        mr_control_parameters_file_name = "/home/user/Documents/simEnv_2018_07_31/ctrl_params_tmp"

        args_list = [force_file_directory + "/simProgram",
                     str(freq), str(sim_time_in_sec),
                     meas_dest_file_name, mr_control_parameters_file_name,
                     "-eF", force_file]

        subprocess.run(args_list)

        with open(force_file_directory + "/file_simResults.txt", 'r') as file:
            content = file.readlines()

        content = [x.strip() for x in content]
        time_stamp_vec = []
        excFreq_vec = []
        l_sim_Xr = [[], [], [], [], [], [], [], [], [], [], [], []]

        for line in content[1:]:
            splitted_line = line.split(';')
            time_stamp_vec.append(splitted_line[0])
            excFreq_vec.append(splitted_line[1])
            for num in range(0, 12):
                l_sim_Xr[num].append(splitted_line[2 + num])

            #and all of the results exctacted from splitted line

        sim_result_dict = dict()
        sim_result_dict['time_stamp'] = time_stamp_vec
        sim_result_dict['excFrea'] = excFreq_vec
        sim_result_dict['l_sim_Xr'] = l_sim_Xr

        sim_results_dict[force_file] = sim_result_dict

        velocity_file_name = write_velocity_to_file(force_file_directory, force_file_name_clean, l_sim_Xr[1])

        velocity_files.append(velocity_file_name)

        print('Next simProgram...')

    return velocity_files


if __name__ == "__main__":
    freq = 500
    sim_time_in_sec = 12
    [force_external_files, force_vectors_dict] = create_force_external_files(freq, sim_time_in_sec)
    velocity_files = run_simPrograms(force_external_files, freq, sim_time_in_sec)

    model_and_inv_model_identification.main(option='u_y_from_file', input_file_list=force_external_files,
                                            output_file_list=velocity_files, title_addon='_MIXED_SINE_TRAINED_')