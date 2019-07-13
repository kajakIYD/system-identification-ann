import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt

import model_and_inv_model_identification as mdl
import simulation
import inertia_modelling
import compare_mses
import calc_R2


def plot_signal(control_full, title='Identification signal'):
    plt.plot(control_full)
    plt.title(title)
    plt.xlabel('Probes')
    plt.grid()
    plt.show()


def plot_identification_model(identification_signal, title_model_data, checkpoint_path,
                              title='Model fit'):
    n_inputs = 1
    n_outputs = 1
    title_model = title_model_data['title']
    n_neurons_model = title_model_data['n_neurons']
    n_steps_model = title_model_data['n_steps']

    init_model, training_op, X_model, y, outputs, loss = inertia_modelling.construct_rnn(n_steps_model, n_inputs,
                                                                                         n_outputs, n_neurons_model)

    saver = tf.train.Saver()

    sess_model = tf.Session()
    init_model.run(session=sess_model)
    saver.restore(sess_model,
                  checkpoint_path)

    previous_output = 0
    reference_output = []
    for i in range(0, len(identification_signal)):
        current_output = inertia_modelling.simulate_step(identification_signal[i], previous_output=previous_output)
        reference_output.append(current_output)
        previous_output = current_output

    control_X_batch_flat = [0] * (n_steps_model - 1) + identification_signal

    output_prediction = []
    for i in range(0, len(identification_signal) - n_steps_model):
        new_temp_x = np.asarray([np.asarray(control_X_batch_flat[i:i + n_steps_model])]).reshape(-1, n_steps_model, 1)
        y_pred_test = sess_model.run(outputs, feed_dict={X_model: new_temp_x})
        output_prediction.append(y_pred_test[-1][-1])

    output_prediction_flat = []
    for i in range(0, len(output_prediction)):
        output_prediction_flat.append(output_prediction[i][0])

    plt.plot(reference_output)
    plt.plot(output_prediction_flat)
    plt.title(title)
    plt.grid()
    plt.show()

    return output_prediction_flat, reference_output


def plot_mse_vs_n_neurons(performances_sorted):
    performances_mse_train_to_plot = dict()
    performances_mse_test_to_plot = dict()

    n_iterations_list = [10, 20, 50]
    n_steps_list = [10, 20, 30, 50]
    n_neurons_list = [50, 200, 500, 1000]

    for n_steps in n_steps_list:
        for n_iterations in n_iterations_list:
            performances_mse_train_to_plot['n_steps_' + str(n_steps) +
                                           '_n_iterations_' + str(n_iterations)] = []
            performances_mse_test_to_plot['n_steps_' + str(n_steps) +
                                           '_n_iterations_' + str(n_iterations)] = []

    for item in performances_sorted:
        n_neurons = item['n_neurons']
        n_steps = item['n_steps']
        n_iterations = item['n_iterations']
        performances_mse_train_to_plot['n_steps_' + str(n_steps) +
                                       '_n_iterations_' + str(n_iterations)].append((item['mse_train_set'], n_neurons))
        performances_mse_test_to_plot['n_steps_' + str(n_steps) +
                                      '_n_iterations_' + str(n_iterations)].append((item['mse_test_set'], n_neurons))

    for n_steps in [10, 50]:
        for n_iterations in [10, 50]:
            mse = [item[0] for item in performances_mse_train_to_plot['n_steps_' + str(n_steps) +
                                                                      '_n_iterations_' + str(n_iterations)]]
            n_neurons_list_to_plot = [item[1] for item in performances_mse_train_to_plot['n_steps_' + str(n_steps) +
                                                                      '_n_iterations_' + str(n_iterations)]]
            plt.plot(n_neurons_list_to_plot, mse, '.', label='n_steps= ' + str(n_steps) + ' n_iterations=' + str(n_iterations))

    plt.xticks([50, 200, 500, 1000])
    plt.yticks(list(plt.yticks()[0]) + [0])
    plt.xlabel('n_neurons')
    plt.ylabel('MSE')
    plt.legend()
    plt.title('MSE_train vs n_neurons')
    plt.show()

    for n_steps in [10, 50]:
        for n_iterations in [10, 50]:
            mse = [item[0] for item in performances_mse_test_to_plot['n_steps_' + str(n_steps) +
                                                                      '_n_iterations_' + str(n_iterations)]]
            n_neurons_list_to_plot = [item[1] for item in performances_mse_test_to_plot['n_steps_' + str(n_steps) +
                                                                      '_n_iterations_' + str(n_iterations)]]
            plt.plot(n_neurons_list_to_plot, mse, '.', label='n_steps= ' + str(n_steps) + ' n_iterations=' + str(n_iterations))

    plt.xticks([50, 200, 500, 1000])
    plt.yticks(list(plt.yticks()[0]) + [0])
    plt.xlabel('n_neurons')
    plt.ylabel('MSE')
    plt.legend()
    plt.title('MSE_test vs n_neurons')
    plt.show()


def plot_mse_vs_n_steps(performances_sorted, title='mse vs n_steps'):
    performances_mse_train_to_plot = dict()
    performances_mse_test_to_plot = dict()

    n_iterations_list = [10, 20, 50]
    n_steps_list = [10, 20, 30, 50]
    n_neurons_list = [50, 200, 500, 1000]

    for n_neurons in n_neurons_list:
        for n_iterations in n_iterations_list:
            performances_mse_train_to_plot['n_neurons_' + str(n_neurons) +
                                           '_n_iterations_' + str(n_iterations)] = []
            performances_mse_test_to_plot['n_neurons_' + str(n_neurons) +
                                          '_n_iterations_' + str(n_iterations)] = []

    for item in performances_sorted:
        n_neurons = item['n_neurons']
        n_steps = item['n_steps']
        n_iterations = item['n_iterations']
        performances_mse_train_to_plot['n_neurons_' + str(n_neurons) +
                                       '_n_iterations_' + str(n_iterations)].append((item['mse_train_set'], n_steps))
        performances_mse_test_to_plot['n_neurons_' + str(n_neurons) +
                                      '_n_iterations_' + str(n_iterations)].append((item['mse_test_set'], n_steps))

    for n_neurons in [50, 1000]:
        for n_iterations in [10, 50]:
            mse = [item[0] for item in performances_mse_train_to_plot['n_neurons_' + str(n_neurons) +
                                                                      '_n_iterations_' + str(n_iterations)]]
            n_neurons_list_to_plot = [item[1] for item in performances_mse_train_to_plot['n_neurons_' + str(n_neurons) +
                                                                                         '_n_iterations_' + str(
                n_iterations)]]
            plt.plot(n_neurons_list_to_plot, mse, '.',
                     label='n_neurons= ' + str(n_neurons) + ' n_iterations=' + str(n_iterations))

    plt.xticks([10, 20, 30, 50])
    plt.yticks(list(plt.yticks()[0]) + [0])
    plt.xlabel('n_steps')
    plt.ylabel('MSE')
    plt.legend()
    plt.title('MSE_train vs n_steps')
    plt.show()

    for n_neurons in [50, 1000]:
        for n_iterations in [10, 50]:
            mse = [item[0] for item in performances_mse_test_to_plot['n_neurons_' + str(n_neurons) +
                                                                     '_n_iterations_' + str(n_iterations)]]
            n_neurons_list_to_plot = [item[1] for item in performances_mse_test_to_plot['n_neurons_' + str(n_neurons) +
                                                                                        '_n_iterations_' + str(n_iterations)]]
            plt.plot(n_neurons_list_to_plot, mse, '.',
                     label='n_neurons= ' + str(n_neurons) + ' n_iterations=' + str(n_iterations))

    plt.xticks([10, 20, 30, 50])
    plt.yticks(list(plt.yticks()[0]) + [0])
    plt.xlabel('n_steps')
    plt.ylabel('MSE')
    plt.legend()
    plt.title('MSE_test vs n_steps')
    plt.show()


def plot_mse_vs_n_iterations(performances_sorted, title='mse vs n_iterations'):
    performances_mse_train_to_plot = dict()
    performances_mse_test_to_plot = dict()

    n_iterations_list = [10, 20, 50]
    n_steps_list = [10, 20, 30, 50]
    n_neurons_list = [50, 200, 500, 1000]

    for n_neurons in n_neurons_list:
        for n_steps in n_steps_list:
            performances_mse_train_to_plot['n_neurons_' + str(n_neurons) +
                                           '_n_steps_' + str(n_steps)] = []
            performances_mse_test_to_plot['n_neurons_' + str(n_neurons) +
                                          '_n_steps_' + str(n_steps)] = []

    for item in performances_sorted:
        n_neurons = item['n_neurons']
        n_steps = item['n_steps']
        n_iterations = item['n_iterations']
        performances_mse_train_to_plot['n_neurons_' + str(n_neurons) +
                                       '_n_steps_' + str(n_steps)].append((item['mse_train_set'], n_iterations))
        performances_mse_test_to_plot['n_neurons_' + str(n_neurons) +
                                      '_n_steps_' + str(n_steps)].append((item['mse_test_set'], n_iterations))

    for n_neurons in [50, 1000]:
        for n_steps in [10, 50]:
            mse = [item[0] for item in performances_mse_train_to_plot['n_neurons_' + str(n_neurons) +
                                                                      '_n_steps_' + str(n_steps)]]
            n_iterations_list_to_plot = [item[1] for item in performances_mse_train_to_plot['n_neurons_' + str(n_neurons) +
                                                                                         '_n_steps_' + str(n_steps)]]
            plt.plot(n_iterations_list_to_plot , mse, '.',
                     label='n_neurons= ' + str(n_neurons) + ' n_steps=' + str(n_steps))

    plt.xticks([10, 20, 50])
    plt.yticks(list(plt.yticks()[0]) + [0])
    plt.xlabel('n_iterations')
    plt.ylabel('MSE')
    plt.legend()
    plt.title('MSE_train vs n_iterations')
    plt.show()

    for n_neurons in [50, 1000]:
        for n_steps in [10, 50]:
            mse = [item[0] for item in performances_mse_test_to_plot['n_neurons_' + str(n_neurons) +
                                                                     '_n_steps_' + str(n_steps)]]
            n_neurons_list_to_plot = [item[1] for item in performances_mse_test_to_plot['n_neurons_' + str(n_neurons) +
                                                                                        '_n_steps_' + str(n_steps)]]
            plt.plot(n_neurons_list_to_plot, mse, '.',
                     label='n_neurons= ' + str(n_neurons) + ' n_steps=' + str(n_steps))

    plt.xticks([10, 20, 50])
    plt.xlabel('n_iterations')
    plt.yticks(list(plt.yticks()[0]) + [0])
    plt.ylabel('MSE')
    plt.legend()
    plt.title('MSE_test vs n_iterations')
    plt.show()


def plot_identification_model_inverse(identification_signal, title_model_inverse_data, checkpoint_path,
                                      title='Inverse model fit'):
    n_inputs = 1
    n_outputs = 1
    title_model_inverse = title_model_inverse_data['title']
    n_neurons_inverse_model = title_model_inverse_data['n_neurons']
    n_steps_inverse_model = title_model_inverse_data['n_steps']


    init_model_inverse, training_op, X_model_inverse, y, outputs_inverse, loss = inertia_modelling.construct_rnn(
        n_steps_inverse_model, n_inputs,
        n_outputs, n_neurons_inverse_model)

    saver = tf.train.Saver()

    sess_model_inverse = tf.Session()
    init_model_inverse.run(session=sess_model_inverse)
    saver.restore(sess_model_inverse,
                  checkpoint_path)

    previous_output = 0
    reference_output = []
    for i in range(0, len(identification_signal)):
        current_output = inertia_modelling.simulate_step(identification_signal[i], previous_output=previous_output)
        reference_output.append(current_output)
        previous_output = current_output

    output_y_batch_flat = [0] * (n_steps_inverse_model - 1) + reference_output

    input_prediction = []
    for i in range(0, len(identification_signal) - n_steps_inverse_model):
        new_temp_y = np.asarray([np.asarray(output_y_batch_flat[i:i + n_steps_inverse_model])]).reshape(-1, n_steps_inverse_model, 1)
        X_pred_test = sess_model_inverse.run(outputs_inverse, feed_dict={X_model_inverse: new_temp_y})
        input_prediction.append(X_pred_test[-1][-1])

    plt.plot(identification_signal)
    plt.plot(input_prediction)
    plt.title(title)
    plt.grid()
    plt.show()

    return input_prediction, identification_signal


def best_mse_test_model_ploting(model_performances_sorted, inverse_model_mode=False,
                                checkpoint_folder='./inertia_modelling_checkpoints_RECTANGLE_A2_P80_dt1_1200probes_TRAINED/'):
    performance_number = 0
    n_neurons, n_steps, n_iterations = simulation.extract_rnn_structure_from_title(
        model_performances_sorted[performance_number]['title'])

    title_model_data = {'title': model_performances_sorted[performance_number]['title'],
                        'n_neurons': n_neurons, 'n_steps': n_steps, 'n_iterations': n_iterations}

    control_full = mdl.generate_identification_signal_2()
    control_full_test = mdl.generate_test_signal()

    if not inverse_model_mode:
        output_prediction, reference_output = plot_identification_model(control_full, title_model_data=title_model_data,
                                              checkpoint_path=checkpoint_folder +
                                                  'my_time_series_model' + title_model_data['title'],
                                              title='Best mse_test_set model fit to train set')
        R2_best_mse_test_train_fit = calc_R2.rSquare(output_prediction, reference_output[:-n_steps])

        output_prediction, reference_output = plot_identification_model(control_full_test, title_model_data=title_model_data,
                                              checkpoint_path=checkpoint_folder +
                                                  'my_time_series_model' + title_model_data['title'],
                                              title='Best mse_test_set model fit to test set')
        R2_best_mse_test_test_fit = calc_R2.rSquare(output_prediction, reference_output[:-n_steps])

        return R2_best_mse_test_train_fit, R2_best_mse_test_test_fit
    else:
        input_prediction, identification_signal = plot_identification_model_inverse(control_full, title_model_inverse_data=title_model_data,
                                                                        checkpoint_path=checkpoint_folder +
                                                                                        'my_time_series_model' +
                                                                                        title_model_data['title'],
                                                                        title='Best mse_test_set model fit to train set')
        R2_best_mse_test_train_fit = calc_R2.rSquare(input_prediction, identification_signal[:-n_steps])

        input_prediction, identification_signal = plot_identification_model_inverse(control_full_test, title_model_inverse_data=title_model_data,
                                                                        checkpoint_path=checkpoint_folder +
                                                                                        'my_time_series_model' +
                                                                                        title_model_data['title'],
                                                                        title='Best mse_test_set model fit to test set')
        R2_best_mse_test_test_fit = calc_R2.rSquare(input_prediction, identification_signal[:-n_steps])

        return R2_best_mse_test_train_fit, R2_best_mse_test_test_fit


def best_mse_train_model_ploting(model_performances_sorted, inverse_model_mode=False,
                                 checkpoint_folder='./inertia_modelling_checkpoints_RECTANGLE_A2_P80_dt1_1200probes_TRAINED/'):
    performance_number = 1
    n_neurons, n_steps, n_iterations = simulation.extract_rnn_structure_from_title(
        model_performances_sorted[performance_number]['title'])

    title_model_data = {'title': model_performances_sorted[performance_number]['title'],
                        'n_neurons': n_neurons, 'n_steps': n_steps, 'n_iterations': n_iterations}

    control_full = mdl.generate_identification_signal_2()
    control_full_test = mdl.generate_test_signal()
    if not inverse_model_mode:
        output_prediction_flat, reference_output = plot_identification_model(control_full, title_model_data=title_model_data,
                                                   checkpoint_path=checkpoint_folder +
                                                   'my_time_series_model' + title_model_data['title'],
                                                   title='Best mse_train_set model fit to train set')

        R2_best_mse_train_train_fit = calc_R2.rSquare(output_prediction_flat, reference_output[:-n_steps])

        output_prediction_flat, reference_output = plot_identification_model(control_full_test, title_model_data=title_model_data,
                                                   checkpoint_path=checkpoint_folder +
                                                   'my_time_series_model' + title_model_data['title'],
                                                   title='Best mse_train_set model fit to test set')

        R2_best_mse_train_test_fit = calc_R2.rSquare(output_prediction_flat, reference_output[:-n_steps])
    else:
        input_prediction, identification_signal = plot_identification_model_inverse(control_full,
                                                                                    title_model_inverse_data=title_model_data,
                                                                                    checkpoint_path=checkpoint_folder +
                                                                                                    'my_time_series_model' +
                                                                                                    title_model_data[
                                                                                                        'title'],
                                                                                    title='Best mse_train_set model inverse_fit to train set')
        R2_best_mse_train_train_fit = calc_R2.rSquare(input_prediction, identification_signal[:-n_steps])

        input_prediction, identification_signal = plot_identification_model_inverse(control_full_test,
                                                                                    title_model_inverse_data=title_model_data,
                                                                                    checkpoint_path=checkpoint_folder +
                                                                                                    'my_time_series_model' +
                                                                                                    title_model_data[
                                                                                                        'title'],
                                                                                    title='Best mse_train_set model_inverse fit to test set')
        R2_best_mse_train_test_fit = calc_R2.rSquare(input_prediction, identification_signal[:-n_steps])

    return R2_best_mse_train_train_fit, R2_best_mse_train_test_fit


def simulate_old_perfect_simulation():
    unpickled_object = compare_mses.unpickle_object(
        "./inertia_modelling_checkpoints/sorted_simulation_performances.pkl")

    compare_mses.simulate_single_simulation(unpickled_object[0], suspension_simulation=False, mixed_simulation=False,
                                            rectangle_simulation=True)

    compare_mses.simulate_single_simulation(unpickled_object[100], suspension_simulation=False, mixed_simulation=False,
                                            rectangle_simulation=True)

    compare_mses.simulate_single_simulation(unpickled_object[200], suspension_simulation=False, mixed_simulation=False,
                                            rectangle_simulation=True)

    compare_mses.simulate_single_simulation(unpickled_object[400], suspension_simulation=False, mixed_simulation=False,
                                            rectangle_simulation=True)


def main():
    # unpickled_object = compare_mses.unpickle_object(
    #     "./model_and_inv_model_identification_mses/model_performance_trying_to_reproduce_MIXED_.pickle")
    #
    # model_performances = []
    # for item in unpickled_object:
    #     model_performances.append(item)

    unpickled_object = compare_mses.unpickle_object(
        "./model_and_inv_model_identification_mses/model_performance_trying_to_reproduce_RECTANGLE_.pickle")

    model_performances = []
    for item in unpickled_object:
        try:
            item['mse_train_set'] = item.pop('mse_training_set')
            model_performances.append(item)
        except:
            model_performances.append(item)

    plot_mse_vs_n_iterations(model_performances)
    plot_mse_vs_n_neurons(model_performances)
    plot_mse_vs_n_steps(model_performances)

    # model_performances_sorted = compare_mses.sort_performanes_by(model_performances, key='mse_test_set')
    model_performances_sorted = compare_mses.sort_performanes_by(model_performances, key='mse_test_set')
    R2_best_mse_test_train_fit, R2_best_mse_test_test_fit = best_mse_test_model_ploting(model_performances_sorted,
                                                                                        inverse_model_mode=True,
                                                                                        checkpoint_folder='./inertia_modelling_checkpoints_MIXED_TRAINED/')
    #
    # print('R2_best_mse_test_train_fit = ' + str(R2_best_mse_test_train_fit))
    # print('R2_best_mse_test_test_fit = ' + str(R2_best_mse_test_test_fit))
    #
    # model_performances_sorted = compare_mses.sort_performanes_by(model_performances, key='mse_training_set')
    model_performances_sorted = compare_mses.sort_performanes_by(model_performances, key='mse_train_set')
    R2_best_mse_train_train_fit, R2_best_mse_train_test_fit = best_mse_train_model_ploting(model_performances_sorted,
                                                                                           inverse_model_mode=True,
                                                                                           checkpoint_folder='./inertia_modelling_checkpoints_MIXED_TRAINED/')
    #
    # print('R2_best_mse_train_train_fit = ' + str(R2_best_mse_train_train_fit))
    # print('R2_best_mse_train_test_fit = ' + str(R2_best_mse_train_test_fit))
    performance_inverse_number = 0
    # n_neurons, n_steps, n_iterations = simulation.extract_rnn_structure_from_title(
    #     model_performances_sorted[performance_inverse_number]['title'])
    #
    # title_model_inverse_data = {'title': model_performances_sorted[performance_inverse_number]['title'], 'n_neurons': n_neurons, 'n_steps': n_steps,
    #                             'n_iterations': n_iterations}
    # plot_identification_model_inverse(control_full, title_model_inverse_data=title_model_inverse_data,
    #                                   checkpoint_path='./inertia_modelling_checkpoints_RECTANGLE_A2_P80_dt1_400probes_TRAINED_old/' +
    #                                                   'my_time_series_model' + title_model_inverse_data['title'])

    # checkpoint_path = "./inertia_modelling_checkpoints"
    # titles_model_inverse_data, titles_model_data = simulation.extract_models_and_inverse_models_data \
    #     (checkpoint_path)
    #
    # plot_identification_model(control_full, title_model_data=titles_model_data[0])
    #
    # plot_identification_model_inverse(control_full, title_model_inverse_data=titles_model_inverse_data[0])

if __name__ == "__main__":
    main()