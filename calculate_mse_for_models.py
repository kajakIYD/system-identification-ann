import tensorflow as tf

import numpy as np
from sklearn.metrics import mean_squared_error
import pickle

import inertia_modelling
import simulation
import model_and_inv_model_identification


def simulate_inertia_for_given_control(control_full):
    output_full = []
    previous_output = 0

    for control in control_full:
        current_output = inertia_modelling.simulate_step(control, previous_output)
        output_full.append(current_output)
        previous_output = current_output

    return output_full


def main(titles_model_inverse_data,
         identification_signal, test_signal,
         title_addon='', training_signal_addon='', option=''):

    output_full = simulate_inertia_for_given_control(identification_signal)
    output_full_test = simulate_inertia_for_given_control(test_signal)

    model_inverse_performance = []

    for title_model_inverse_data in titles_model_inverse_data:
        n_inputs = 1
        n_outputs = 1

        n_steps_inverse_model = title_model_inverse_data['n_steps']
        n_iterations_model_inverse = title_model_inverse_data['n_iterations']
        n_neurons_inverse_model = title_model_inverse_data['n_neurons']
        title_model_inverse = title_model_inverse_data['title']

        init_model_inverse, training_op, X_model_inverse, y, outputs_inverse, loss = inertia_modelling.construct_rnn(n_steps_inverse_model, n_inputs,
                                                                                         n_outputs, n_neurons_inverse_model)

        saver = tf.train.Saver()

        with tf.Session() as sess_model_inverse:  # not shown in the book
            init_model_inverse.run(session=sess_model_inverse)
            saver.restore(sess_model_inverse,
                          title_model_inverse)

            identification_signal_new = [0] * (n_steps_inverse_model - 1) + identification_signal
            output_full_new = [0] * (n_steps_inverse_model - 1) + output_full

            print("Inverse model " + "./" + title_model_inverse + "/my_time_series_model"
                  + " train")

            # Testowanie na zbiorze uczącym
            input_prediction_train = []
            for i in range(0, len(output_full_new) - n_steps_inverse_model):
                new_temp_y = np.asarray([np.asarray(output_full_new[i:i + n_steps_inverse_model])]).reshape(-1, n_steps_inverse_model, 1)
                X_pred = sess_model_inverse.run(outputs_inverse, feed_dict={X_model_inverse: new_temp_y})
                input_prediction_train.append(X_pred[-1][-1])

            test_signal_new = [0] * (n_steps_inverse_model - 1) + test_signal
            output_full_test_new = [0] * (n_steps_inverse_model - 1) + output_full_test

            # Testowanie na zbiorze testowym
            input_prediction_test = []
            for i in range(0, len(output_full_test_new) - n_steps_inverse_model):
                new_temp_y = np.asarray([np.asarray(output_full_test_new[i:i + n_steps_inverse_model])]).reshape(-1, n_steps_inverse_model, 1)
                X_pred_test = sess_model_inverse.run(outputs_inverse, feed_dict={X_model_inverse: new_temp_y})
                input_prediction_test.append(X_pred_test[-1][-1])

            input_prediction_train = [subitem for sublist in input_prediction_train for subitem in sublist]
            input_prediction_test = [subitem for sublist in input_prediction_test for subitem in sublist]

            try:
                mse_train_set = mean_squared_error(identification_signal[:-1],
                                                   input_prediction_train)
            except ValueError:
                mse_train_set = 0.0

            try:
                mse_test_set = mean_squared_error(test_signal[:-1], input_prediction_test)
            except ValueError:
                mse_test_set = 0.0

            model_inverse_performance.append(
                {
                    'mse_train_set': mse_train_set,
                    'mse_test_set': mse_test_set,
                    'n_steps': n_steps_inverse_model, 'n_neurons': n_neurons_inverse_model, 'n_iterations': n_iterations_model_inverse,
                    'title': title_model_inverse,
                    'identification_option': option, 'training_signal_addon': training_signal_addon
                })

            with open(
                    r"model_and_inv_model_identification_mses/model_inverse_performance" + title_addon
                    + option + ".pickle",
                    "wb") as output_file:
                pickle.dump(model_inverse_performance, output_file)


if __name__ == "__main__":
    titles_model_inverse_data, titles_model_data = simulation.extract_models_and_inverse_models_data\
                                                    ("./inertia_modelling_checkpoints_MIXED_TRAINED")

    identification_signal = model_and_inv_model_identification.generate_identification_signal_2()
    test_signal = model_and_inv_model_identification.generate_test_signal()

    main(titles_model_inverse_data,
         identification_signal, test_signal,
         title_addon='_MIXED_TRAINED_', training_signal_addon='', option='inertia_modelling')