import tensorflow as tf

import numpy as np
import matplotlib.pyplot as plt

import inertia_modelling

def main():
    dt = 0.1

    SP = 1

    n_steps_model = 50
    n_steps_inverse_model = 20

    n_inputs = 1
    n_outputs = 1
    n_neurons_model = 500
    n_neurons_inverse_model = 200

    init_model_inverse, training_op, X_model_inverse, y, outputs_inverse, loss = inertia_modelling.construct_rnn(n_steps_inverse_model, n_inputs,
                                                                                     n_outputs, n_neurons_inverse_model)

    saver = tf.train.Saver()

    sess_model_inverse = tf.Session()
    init_model_inverse.run(session=sess_model_inverse)
    saver.restore(sess_model_inverse,
                  "./inertia_modelling_checkpoints/my_time_series_modelINVERSE_n_iterations_20 n_steps_20 n_neurons_200")  # not shown

    init_model, training_op, X_model, y, outputs, loss = inertia_modelling.construct_rnn(n_steps_model, n_inputs,
                                                                             n_outputs, n_neurons_model)

    saver = tf.train.Saver()

    sess_model = tf.Session()
    init_model.run(session=sess_model)
    saver.restore(sess_model,
                  "./inertia_modelling_checkpoints/my_time_series_modeln_iterations_250 n_steps_50 n_neurons_500")  # not shown

    previous_model_plant_disturbed_difference = [0] * n_steps_inverse_model
    inverse_model_input_vector = [0] * n_steps_inverse_model
    model_input_vector = [0] * n_steps_model

    plant_output = [0]
    plant_control = []
    disturbed_plant_output = []
    model_plant_disturbed_difference = []
    SP_feedback_difference = []

    simulation_time = 10
    loop_counter = 1

    # inertia parameters
    a = 1
    b = 0.1

    for t in range(0, int(simulation_time / dt)):
        inverse_model_input_vector[-1] = SP - previous_model_plant_disturbed_difference[-1]
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
    plt.show()


if __name__ == "__main__":
    main()