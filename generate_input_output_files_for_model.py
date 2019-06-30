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


def run_simPrograms(force_external_files, freq, sim_time_in_sec, force_vectors_dict, plotting=True):
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

        subprocess.call(args_list)

        time.sleep(0.1)

        with open(force_file_directory + "/file_simResults.txt", 'r') as file:
            content = file.readlines()

        content = [x.strip() for x in content]
        time_stamp_vec = []
        excFreq_vec = []
        roadExc_vec = [[], [], [], [], [], [], [], [], [], [], [], []]
        # 0:7 disp 8:15 vel
        l_sim_fullCar_Y = [[], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], []]

        for line in content[1:]:
            splitted_line = line.split(';')
            time_stamp_vec.append(float(splitted_line[0]))
            excFreq_vec.append(float(splitted_line[1]))
            for num in range(0, 12):
                roadExc_vec[num].append(float(splitted_line[2 + num]))
            for num in range(0, 24):
                l_sim_fullCar_Y[num].append(float(splitted_line[14 + num]))

            #and all of the results exctacted from splitted line

        sim_result_dict = dict()
        sim_result_dict['time_stamp'] = time_stamp_vec
        sim_result_dict['excFreq'] = excFreq_vec
        sim_result_dict['roadExc'] = roadExc_vec
        sim_result_dict['l_sim_fullCar_Y'] = l_sim_fullCar_Y

        sim_results_dict[force_file] = sim_result_dict

        velocity_file_name = write_velocity_to_file(force_file_directory, force_file_name_clean, l_sim_fullCar_Y[14])

        velocity_files.append(velocity_file_name)

        if plotting:
            fig, ax1 = plt.subplots()
            ax1.plot(l_sim_fullCar_Y[14], 'b-')
            ax1.set_xlabel('Probes')
            # Make the y-axis label, ticks and tick labels match the line color.
            ax1.set_ylabel('velocity, m/s', color='b')
            ax1.tick_params('y', colors='b')
            ax2 = ax1.twinx()
            ax2.plot(force_vectors_dict[force_file], 'r')
            ax2.set_ylabel('Control force, N', color='r')
            ax2.tick_params('y', colors='r')

            fig.tight_layout()
            plt.title(force_file)
            plt.show()


        print('Next simProgram...')

    simulation.pickle_object(sim_results_dict, "sim_results_dict.pkl")

    return velocity_files


def compile_proper_simulator_in_non_control_mode(mode='active_suspension'):
    # mode='semi_active_suspension'

    # make change in configuration.h (enable TCP mode and proper suspension model)
    fileName = '/home/user/Documents/simEnv_2018_07_31/configuration.h'

    with open(fileName, "r") as file:
        content = file.read()

    content_modified = ''
    for line in content.split('\n'):
        if r"#define TCP_ONLINE_SIMULATION" in line and r"//#define TCP_ONLINE_SIMULATION" not in line:
            line = line.replace(r"#define TCP_ONLINE_SIMULATION", r"//#define TCP_ONLINE_SIMULATION")
        if r"//#define FULL_ACTIVE_SUSPENSION_IDENTIFICATION" in line:
            line = line.replace(r"//#define FULL_ACTIVE_SUSPENSION_IDENTIFICATION", r"#define FULL_ACTIVE_SUSPENSION_IDENTIFICATION")
        if r"//#define ROAD_EXC_OFF" in line:
            line = line.replace(r"//#define ROAD_EXC_OFF", r"#define ROAD_EXC_OFF")
        if r"#define SIN_FREQ_ROAD_EXC" in line and r"//#define SIN_FREQ_ROAD_EXC" not in line:
            line = line.replace(r"#define SIN_FREQ_ROAD_EXC", r"//#define SIN_FREQ_ROAD_EXC")
        if r"//#define SAVE_RESULTS" in line:
            line = line.replace(r"//#define SAVE_RESULTS", r"#define SAVE_RESULTS")
        if r"#define NO_BYPASS_CONTROL" in line and r"//#define NO_BYPASS_CONTROL" not in line:
            line = line.replace(r"#define NO_BYPASS_CONTROL", r"//#define NO_BYPASS_CONTROL")

        content_modified = content_modified + line + '\n'

    with open(fileName, "w") as file:
        file.write(content_modified)

    # make change in Makefile
    fileName = '/home/user/Documents/simEnv_2018_07_31/Makefile'

    with open(fileName, "r") as file:
        content = file.read()

    content_modified = ''
    for line in content.split('\n'):
        if r"#ROAD_EXC = roadExcOff" in line:
            line = line.replace(r"#ROAD_EXC = roadExcOff", r"ROAD_EXC = roadExcOff")
        if r"ROAD_EXC = sinFreqRoadExc" in line and r"#ROAD_EXC = sinFreqRoadExc" not in line:
            line = line.replace(r"ROAD_EXC = sinFreqRoadExc", r"#ROAD_EXC = sinFreqRoadExc")

        content_modified = content_modified + line + '\n'

    with open(fileName, "w") as file:
        file.write(content_modified)

    # make clean
    args_list = ['make', 'clean', '-C', '/home/user/Documents/simEnv_2018_07_31']
    subprocess.run(args_list)

    # make all
    args_list = ['make', 'all', '-C', '/home/user/Documents/simEnv_2018_07_31']
    subprocess.run(args_list)


if __name__ == "__main__":
    freq = 500
    sim_time_in_sec = 30
    [force_external_files, force_vectors_dict] = create_force_external_files(freq, sim_time_in_sec, scenarios_cnt=1)
    compile_proper_simulator_in_non_control_mode(mode='active_suspension')
    velocity_files = run_simPrograms(force_external_files, freq, sim_time_in_sec, force_vectors_dict, plotting=False)

    print("FINISHED generate_input+output_files_for_model!!!!!")

    model_and_inv_model_identification.main(option='u_y_from_file', input_file_list=force_external_files,
                                            output_file_list=velocity_files, title_addon='_MIXED_SINE_TRAINED_',
                                            ploting=False)

    print("FINISHED perform indentification in generate_input_output_files_for_model!!!!!")
