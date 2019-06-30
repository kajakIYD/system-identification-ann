# Different controls applied to perform model and inverse model identification
import math
import matplotlib.pyplot as plt
import random

import inertia_modelling
import simulation


def generate_step(time, amplitude):
    return [amplitude] * time


def generate_sine(time, amplitude, omega, fi=0):
    return [amplitude * math.sin(omega * t) for t in range(time)]


def generate_rectangle(time, amplitude, period, dt=1):
    output = []
    for t in range(int(time / dt)):
        if t % (2 * period / dt) < period / dt:
            output.append(amplitude)
        else:
            output.append(-amplitude)

    return output


def main(option='u_y_from_file', input_file_list=[], output_file_list=[], title_addon='SINGLE_SINE_TRAINED',
         ploting=False):

    if option == 'inertia_modelling':
        full_experiment_length = 0

        experiment_length_part = 400
        amplitude = 3
        omega = 0.1
        control_full = generate_sine(experiment_length_part, amplitude, omega)

        full_experiment_length = full_experiment_length + experiment_length_part

        # experiment_length_part = 500
        # amplitude = 2
        # period = 100
        # control_full = control_full + generate_rectangle(experiment_length_part, amplitude, period)
        #
        # full_experiment_length = full_experiment_length + experiment_length_part
        #
        # experiment_length_part = 400
        # amplitude = 1
        # omega = 0.05
        # control_full = control_full + generate_sine(experiment_length_part, amplitude, omega)
        #
        # full_experiment_length = full_experiment_length + experiment_length_part
        #
        # experiment_length_part = 200
        # amplitude = -3
        # control_full = control_full + generate_step(experiment_length_part, amplitude)
        #
        # full_experiment_length = full_experiment_length + experiment_length_part

        # experiment_length_part = 700
        # amplitude = 1
        # omega = 0.01
        # control_full = control_full + generate_sine(experiment_length_part, amplitude, omega)
        #
        # full_experiment_length = full_experiment_length + experiment_length_part
        #
        # experiment_length_part = 200
        # amplitude = 1
        # control_full = control_full + generate_step(experiment_length_part, amplitude)
        #
        # full_experiment_length = full_experiment_length + experiment_length_part
        #
        # experiment_length_part = 600
        # amplitude = 0.5
        # period = 200
        # control_full = control_full + generate_rectangle(experiment_length_part, amplitude, period)
        #
        # full_experiment_length = full_experiment_length + experiment_length_part

        # plt.plot(range(full_experiment_length), control_full)
        # plt.title("Whole control that will be aplied to experiment")
        # plt.show(block=False) # but this option closes window immediately

        control_full_test = [item + random.uniform(-0.2, 0.2) for item in control_full[::-1]]

        inertia_modelling.perform_identification(control_full, full_experiment_length, control_full_test, "_MIXED_",
                                                 ploting=True)
    elif option == 'u_y_from_file':

        if len(input_file_list) == 0 and len(output_file_list) == 0:
            input_file_list = ["/home/user/Documents/simEnv_2018_07_31/forceExternal_len_200000_1.txt"]
            output_file_list = ["/home/user/Documents/simEnv_2018_07_31/vel_1_len_200000_1.txt"]

        for input_file, output_file in zip(input_file_list, output_file_list):
            input_file_title = input_file.split('/')[-1][:-4]
            output_file_title = output_file.split('/')[-1][:-4]

            control_full = []
            with open(input_file) as f:
                for line in f:
                    control_full.append(float(line.rstrip('\n')))

            output_full = []
            with open(output_file) as f:
                for line in f:
                    output_full.append(float(line.rstrip('\n')))

            full_experiment_length = len(control_full)
            control_full_test = [item + random.uniform(-0.2, 0.2) for item in control_full[::-1]]
            output_full_test = output_full  # TODO!!! OUTPUT CALCULATED USING DISTURBED INPUT!!!!

            inertia_modelling.perform_identification(control_full, full_experiment_length, control_full_test,
                                                     title_addon=title_addon + input_file_title + "_" + output_file_title,
                                                     option=option, output_full=output_full,
                                                     output_full_test=output_full_test, ploting=ploting)

    print("FINISHED model_and_inv_model_identification!!!!!")


if __name__ == "__main__":
    main(option='inertia_modelling')
    # main(option='u_y_from_file')