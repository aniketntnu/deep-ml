import numpy as np

class LSTM:

    def __init__(self, input_size, hidden_size):
        """
        Initialize the LSTM parameters.

        Parameters
        ----------
        input_size : Number of features in each input vector.
        hidden_size : Number of hidden units (memory cells).
        """

        self.input_size = input_size
        self.hidden_size = hidden_size

        # -------------------------------------------------------
        # Weight matrices
        #
        # Every gate receives
        #
        # [h(t-1), x(t)]
        #
        # Therefore input dimension is
        #
        # hidden_size + input_size
        # -------------------------------------------------------

        self.Wf = np.random.randn(hidden_size, hidden_size + input_size)
        # Forget gate weights

        self.Wi = np.random.randn(hidden_size, hidden_size + input_size)
        # Input gate weights

        self.Wc = np.random.randn(hidden_size, hidden_size + input_size)
        # Candidate cell weights

        self.Wo = np.random.randn(hidden_size, hidden_size + input_size)
        # Output gate weights

        # -------------------------------------------------------
        # Bias vectors
        # -------------------------------------------------------

        self.bf = np.zeros((hidden_size, 1))
        self.bi = np.zeros((hidden_size, 1))
        self.bc = np.zeros((hidden_size, 1))
        self.bo = np.zeros((hidden_size, 1))

    # -----------------------------------------------------------
    # Sigmoid activation
    # -----------------------------------------------------------

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    # -----------------------------------------------------------
    # Forward Pass
    # -----------------------------------------------------------

    def forward(self,
                x,
                initial_hidden_state,
                initial_cell_state):
        """
        Parameters
        ----------
        x : ndarray
            Shape = (sequence_length, input_size)

        initial_hidden_state :
            Shape = (hidden_size,1)

        initial_cell_state :
            Shape = (hidden_size,1)

        Returns
        -------
        outputs :
            Hidden state at every time step.

        h :
            Final hidden state.

        c :
            Final cell state.
        """

        # Initialize hidden state

        h = initial_hidden_state.copy()

        # Initialize cell state

        c = initial_cell_state.copy()

        # Store hidden states

        outputs = []

        # -------------------------------------------------------
        # Process sequence one timestep at a time
        # -------------------------------------------------------

        for t in range(len(x)):

            # Current input

            xt = x[t].reshape(-1, 1)

            # Concatenate previous hidden state
            # with current input

            concat = np.vstack((h, xt))

            # ===================================================
            # Forget Gate
            # ===================================================

            f = self.sigmoid(
                np.dot(self.Wf, concat) + self.bf
            )

            # ===================================================
            # Input Gate
            # ===================================================

            i = self.sigmoid(
                np.dot(self.Wi, concat) + self.bi
            )

            # ===================================================
            # Candidate Cell State
            # ===================================================

            c_candidate = np.tanh(
                np.dot(self.Wc, concat) + self.bc
            )

            # ===================================================
            # Update Cell State
            # ===================================================

            c = f * c + i * c_candidate

            # ===================================================
            # Output Gate
            # ===================================================

            o = self.sigmoid(
                np.dot(self.Wo, concat) + self.bo
            )

            # ===================================================
            # Hidden State
            # ===================================================

            h = o * np.tanh(c)

            # Store hidden state

            outputs.append(h.copy())

        # Convert list into numpy array

        outputs = np.array(outputs)

        return outputs, h, c