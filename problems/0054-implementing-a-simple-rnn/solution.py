import numpy as np


def rnn_forward(
    input_sequence: list[list[float]],
    initial_hidden_state: list[float],
    Wx: list[list[float]],
    Wh: list[list[float]],
    b: list[float]
) -> list[float]:

    # Convert everything to NumPy arrays
    input_sequence = np.array(input_sequence)
    h = np.array(initial_hidden_state)
    Wx = np.array(Wx)
    Wh = np.array(Wh)
    b = np.array(b)

    # Process one input at a time
    for x in input_sequence:

        # RNN equation:
        # h_new = tanh(Wx*x + Wh*h + b)
        h = np.tanh(Wx @ x + Wh @ h + b)

    # Round the final hidden state to 4 decimal places
    final_hidden_state = np.round(h, 4)

    # Convert NumPy array to Python list
    return final_hidden_state.tolist()


# Example
input_sequence = [[1.0], [2.0], [3.0]]
initial_hidden_state = [0.0]

Wx = [[0.5]]
Wh = [[0.8]]
b = [0.0]

final_hidden_state = rnn_forward(
    input_sequence,
    initial_hidden_state,
    Wx,
    Wh,
    b
)

print(final_hidden_state)