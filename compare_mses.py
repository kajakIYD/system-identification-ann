#Compare mses from pickles
from pathlib import Path
import pickle
import fnmatch
import os

import simulation
import model_and_inv_model_identification


def unpickle_object(path):
    with open(path, "rb") as input_file:
        object_to_unpickle = pickle.load(input_file)

    return object_to_unpickle


def sort_performanes_by(performances, key='mse'):
    performances_flat = []
    mses_vals = []

    for item in performances:
        if len(item) > 0:
            mses_vals.append(item[key])
            performances_flat.append(item)

    sorted_performances = sorted(performances_flat, key=lambda k: k['mse'])
    return sorted_performances


# Check if best identification performance means best simulation performance
def compare_best_identification_vs_best_simulation_performance(identification_performance_filepath,
                                                               simulation_performance_filepath):
    ident_perf_dict = sort_performanes_by(unpickle_object(identification_performance_filepath))
    sim_perf_dict = sort_performanes_by(unpickle_object(simulation_performance_filepath))

    for y in ident_perf_dict:
        print(y, ':', ident_perf_dict[y])

    for y in sim_perf_dict:
        print(y, ':', ident_perf_dict[y])


def extract_identification_performances(directory_in_str="."):
    identification_performances = []

    for file in os.listdir(directory_in_str):
        if fnmatch.fnmatch(file, 'mses_*'):
            identification_performances.append(unpickle_object(directory_in_str + "/" + str(file)))

    return identification_performances


def extract_simulation_performances(directory_in_str="."):
    simulation_performances = []

    for file in os.listdir(directory_in_str):
        if fnmatch.fnmatch(file, 'mses_*'):
            unpickled_object = unpickle_object(directory_in_str + "/" + str(file))
            for item in unpickled_object:
                simulation_performances.append(item)

    return simulation_performances


def simulate_single_simulation(single_simulation_dict, suspension_simulation=True):
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

    if suspension_simulation:
        simulation.simulation_TCP([title_model_inverse_data], [title_model_data], mse_calc=False,  plotting=True)
    else:
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


def analyze_identification_mses_and_run_simulations():
    identification_performances = extract_identification_performances(directory_in_str="./inertia_modelling_performances")

    mses_vals = []

    identification_performances_flat = []

    for item in identification_performances:
        if len(item) > 0:
            item = item[0]
            mses_vals.append(item['mse'])
            identification_performances_flat.append(item)

    sorted_identification_performances = sorted(identification_performances_flat, key=lambda k: k['mse'])

    min_mse_index = mses_vals.index(min(mses_vals))

    best_identification = identification_performances_flat[min_mse_index]

    print(best_identification)

    simulation.pickle_object(best_identification, "best_identification.pkl")
    simulation.pickle_object(sorted_identification_performances, "sorted_modelling_performances.pkl")

    # simulate_single_simulation(best_simulation, suspension_simulation=True)
    #
    # simulate_single_simulation(sorted_identification_performances[10], suspension_simulation=True)
    # #
    # simulate_single_simulation(sorted_identification_performances[50], suspension_simulation=True)
    #
    simulate_single_simulation(sorted_identification_performances[100], suspension_simulation=True)
    #
    simulate_single_simulation(sorted_identification_performances[200], suspension_simulation=True)
    #
    # simulate_single_simulation(sorted_identification_performances[500], suspension_simulation=True)

    simulate_single_simulation(sorted_identification_performances[-10], suspension_simulation=True)

    simulate_single_simulation(sorted_identification_performances[-1], suspension_simulation=True)
    #
    # simulate_single_simulation(sorted_simulation_performances[-100])


def analyze_simulation_mses_and_run_simulations():
    simulation_performances = extract_simulation_performances(directory_in_str="./active_suspension_simulation_performances")

    mses_vals = []

    simulation_performances_flat = []

    for item in simulation_performances:
        if len(item) > 0:
            mses_vals.append(item['mse'])
            simulation_performances_flat.append(item)

    sorted_simulation_performances = sorted(simulation_performances_flat, key=lambda k: k['mse'])

    min_mse_index = mses_vals.index(min(mses_vals))

    best_simulation = simulation_performances_flat[min_mse_index]

    print(best_simulation)

    simulation.pickle_object(best_simulation, "best_simulation.pkl")
    simulation.pickle_object(sorted_simulation_performances, "sorted_simulation_performances.pkl")

    simulate_single_simulation(best_simulation, suspension_simulation=True)

    simulate_single_simulation(sorted_simulation_performances[10], suspension_simulation=True)

    simulate_single_simulation(sorted_simulation_performances[-10], suspension_simulation=True)

    simulate_single_simulation(sorted_simulation_performances[-1], suspension_simulation=True)
    #
    # simulate_single_simulation(sorted_simulation_performances[-100])


def main():
    # analyze_identification_mses_and_run_simulations()
    # analyze_simulation_mses_and_run_simulations()
    compare_best_identification_vs_best_simulation_performance('sorted_modelling_performances.pkl',
                                                               './inertia_simulation_performances/mses_3600.pkl')


if __name__ == "__main__":
    main()