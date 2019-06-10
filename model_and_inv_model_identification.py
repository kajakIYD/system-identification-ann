# Different controls applied to perform model and inverse model identification
import math
import matplotlib.pyplot as plt

import inertia_modelling
import simulation


def generate_step(time, amplitude):
    return [amplitude] * time


def generate_sine(time, amplitude, omega, fi=0):
    return [amplitude * math.sin(omega * t) for t in range(time)]


def generate_rectangle(time, amplitude, period):
    output = []
    for t in range(time):
        if t % (2 * period) < period:
            output.append(amplitude)
        else:
            output.append(-amplitude)

    return output


def main():
    full_experiment_length = 0

    experiment_length_part = 400
    amplitude = 3
    omega = 0.1
    control_full = generate_sine(experiment_length_part, amplitude, omega)

    full_experiment_length = full_experiment_length + experiment_length_part

    experiment_length_part = 500
    amplitude = 2
    period = 100
    control_full = control_full + generate_rectangle(experiment_length_part, amplitude, period)

    full_experiment_length = full_experiment_length + experiment_length_part

    experiment_length_part = 400
    amplitude = 1
    omega = 0.05
    control_full = control_full + generate_sine(experiment_length_part, amplitude, omega)

    full_experiment_length = full_experiment_length + experiment_length_part

    experiment_length_part = 200
    amplitude = -3
    control_full = control_full + generate_step(experiment_length_part, amplitude)

    full_experiment_length = full_experiment_length + experiment_length_part

    experiment_length_part = 700
    amplitude = 1
    omega = 0.01
    control_full = control_full + generate_sine(experiment_length_part, amplitude, omega)

    full_experiment_length = full_experiment_length + experiment_length_part

    experiment_length_part = 200
    amplitude = 1
    control_full = control_full + generate_step(experiment_length_part, amplitude)

    full_experiment_length = full_experiment_length + experiment_length_part

    experiment_length_part = 600
    amplitude = 0.5
    period = 200
    control_full = control_full + generate_rectangle(experiment_length_part, amplitude, period)

    full_experiment_length = full_experiment_length + experiment_length_part

    plt.plot(range(full_experiment_length), control_full)
    plt.title("Whole control that will be aplied to experiment")
    plt.show()

    control_full_test = control_full[::-1]

    inertia_modelling.perform_identification(control_full, full_experiment_length, control_full_test, "_MIXED_")


if __name__ == "__main__":
    main()