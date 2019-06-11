import tensorflow as tf

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

import inertia_modelling

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



    plt.rcParams.update({'font.size': 6})

    simulation_time = 30
    dt = 0.1

    SP = []
    SP = [1] * int(simulation_time / (2 * dt)) + [2] * int(simulation_time / (2 * dt))

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

                plant_control.append(y_pred_inverse_model[-1][-1])
                plant_output.append(inertia_modelling.simulate_step(y_pred_inverse_model[-1][-1],
                                                                    plant_output[loop_counter - 1], a, b))

                disturbances = 0
                disturbed_plant_output.append(plant_output[loop_counter] + disturbances)

                previous_model_plant_disturbed_difference = previous_model_plant_disturbed_difference[1:]
                previous_model_plant_disturbed_difference.append(disturbed_plant_output[loop_counter - 1] - y_pred_model[-1][-1])

                inverse_model_input_vector = inverse_model_input_vector[1:]
                inverse_model_input_vector.append(0)

                loop_counter = loop_counter + 1

                if loop_counter % 100 == 0:
                    print(str(loop_counter) + "\n")

            loop_counter = loop_counter - 1

            plt.plot(range(0, loop_counter), disturbed_plant_output, "bo", label="Disturbed plant output")
            plt.plot(range(0, loop_counter), plant_control, "ro", label="Inverse model output (control)")
            plt.plot(range(0, loop_counter), model_plant_disturbed_difference, "g.", label="Disturbed plant - model_output")
            plt.plot(range(0, loop_counter), SP_feedback_difference, "m.", label="SP_feedback_difference")
            plt.legend()
            plt.xlabel("Time")
            plt.title("model_inverse: neurons" + str(n_neurons_inverse_model) + " steps" + str(n_steps_inverse_model)
                      + " n_iterations:" + str(n_iterations_model_inverse) + "model: neurons "
                      + str(n_neurons_model) + " steps" + str(n_steps_inverse_model)
                      + " n_iterations:" + str(n_iterations_model))
            plt.show()


if __name__ == "__main__":
    main()