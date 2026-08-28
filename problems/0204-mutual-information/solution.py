import numpy as np

def mutual_information(joint_prob: list[list[float]]) -> float:
    """
    Compute the mutual information between two random variables.

    Args:
        joint_prob: 2D joint probability distribution P(X,Y)

    Returns:
        Mutual information I(X;Y)
    """

    # Convert input into a NumPy array
    joint_prob = np.array(joint_prob, dtype=float)

    # ---------------------------------------------------------
    # Calculate marginal probability P(X)
    # ---------------------------------------------------------
    #
    # Sum across each row.
    #
    # axis=1 means:
    # sum across columns
    #
    # Example:
    #
    # [0.4, 0.1] -> 0.5
    # [0.1, 0.4] -> 0.5
    #
    px = np.sum(joint_prob, axis=1)

    # ---------------------------------------------------------
    # Calculate marginal probability P(Y)
    # ---------------------------------------------------------
    #
    # Sum down each column.
    #
    # axis=0 means:
    # sum down rows
    #
    # Example:
    #
    # column 1: 0.4 + 0.1 = 0.5
    # column 2: 0.1 + 0.4 = 0.5
    #
    py = np.sum(joint_prob, axis=0)

    # ---------------------------------------------------------
    # Calculate P(X)P(Y) for every combination of X and Y
    # ---------------------------------------------------------
    #
    # px[:, None] changes
    #
    # [0.5, 0.5]
    #
    # into
    #
    # [[0.5],
    #  [0.5]]
    #
    # Then multiplying by py gives:
    #
    # [[0.25, 0.25],
    #  [0.25, 0.25]]
    #
    px_py = px[:, None] * py

    # ---------------------------------------------------------
    # Avoid log(0)
    # ---------------------------------------------------------
    #
    # If a joint probability is zero, then
    #
    # P(X,Y) log(P(X,Y)/(P(X)P(Y)))
    #
    # contributes 0 to mutual information.
    #
    # So we only calculate terms where joint_prob > 0.
    #
    mask = joint_prob > 0

    # ---------------------------------------------------------
    # Calculate mutual information
    # ---------------------------------------------------------

    mi = np.sum(
        joint_prob[mask]
        * np.log(
            joint_prob[mask] / px_py[mask]
        )
    )

    # Return as a normal Python float
    return float(mi)