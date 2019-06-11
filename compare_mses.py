#Compare mses from pickles
from pathlib import Path
import pickle
import fnmatch
import os

import simulation
import model_and_inv_model_identification


def unpickle_object(path):
    with open(path, "rb") as input_file:
        simulation_performance = pickle.load(input_file)

    return simulation_performance


def extract_simulation_performances(directory_in_str="./"):
    simulation_performances = []

    for file in os.listdir('.'):
        if fnmatch.fnmatch(file, 'mses_*'):
            simulation_performances.append(unpickle_object(str(file)))

    return simulation_performances


def simulate_single_simulation(single_simulation_dict):
    title = single_simulation_dict['model_inverse_title']
    n_neurons, n_steps, n_iterations = simulation.extract_rnn_structure_from_title(title)

    title_model_inverse_data = {
        'title': title, 'n_neurons': n_neurons,
        'n_steps': n_steps, 'n_iterations': n_iterations
    }

    title = single_simulation_dict['model_title']
    n_neurons, n_steps, n_iterations = simulation.extract_rnn_structure_from_title(title)
    title_model_data = {
        'title': title, 'n_neurons': n_neurons,
        'n_steps': n_steps, 'n_iterations': n_iterations
    }

    simulation.main([title_model_inverse_data], [title_model_data], mse_calc=False, plotting=True)

    simulation_time = 40
    dt = 0.5
    amplitude = 2
    period = 10
    simulation.main([title_model_inverse_data], [title_model_data], simulation_time=simulation_time, dt=dt,
                    SP=model_and_inv_model_identification.generate_rectangle(simulation_time, amplitude,
                                                                             period, dt=dt),
                    mse_calc=False, plotting=True)

    simulation_time = 100
    period = 70
    simulation.main([title_model_inverse_data], [title_model_data], simulation_time=simulation_time, dt=dt,
                    SP=model_and_inv_model_identification.generate_rectangle(simulation_time, amplitude,
                                                                             period, dt=dt),
                    mse_calc=False, plotting=True)


def main():
    simulation_performances = extract_simulation_performances()

    mses_vals = []

    simulation_performances_flat = []

    for item in simulation_performances:
        if len(item) > 0:
            item = item[0]
            mses_vals.append(item['mse'])
            simulation_performances_flat.append(item)

    sorted_simulation_performances = sorted(simulation_performances_flat, key=lambda k: k['mse'])

    min_mse_index = mses_vals.index(min(mses_vals))

    best_simulation = simulation_performances_flat[min_mse_index]

    print(best_simulation)

    simulation.pickle_object(best_simulation, "mses.pkl")
    simulation.pickle_object(sorted_simulation_performances, "sorted_simulation_performances.pkl")

    simulate_single_simulation(best_simulation)

    # simulate_single_simulation(sorted_simulation_performances[10])
    #
    # simulate_single_simulation(sorted_simulation_performances[50])
    #
    # simulate_single_simulation(sorted_simulation_performances[100])
    #
    # simulate_single_simulation(sorted_simulation_performances[200])
    #
    # simulate_single_simulation(sorted_simulation_performances[500])
    #
    # simulate_single_simulation(sorted_simulation_performances[-100])

if __name__ == "__main__":
    main()