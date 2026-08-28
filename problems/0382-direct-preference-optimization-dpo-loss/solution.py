import numpy as np


def dpo_loss(log_probs_chosen_policy: list, log_probs_rejected_policy: list,
             log_probs_chosen_ref: list, log_probs_rejected_ref: list,
             beta: float) -> dict:

    # Our example has 2 samples:
    #
    # chosen_policy   = [-2.0, -1.5]
    # rejected_policy = [-3.0, -2.5]
    #
    # chosen_ref      = [-2.5, -2.0]
    # rejected_ref    = [-2.5, -2.0]
    #
    # beta = 0.5


    # Convert the chosen-policy list and chosen-reference list
    # into NumPy arrays so we can subtract them element by element.
    #
    # Example:
    # [-2.0, -1.5] - [-2.5, -2.0]
    #
    # = [-2.0 - (-2.5), -1.5 - (-2.0)]
    #
    # = [0.5, 0.5]
    #
    # This tells us how much the policy has increased its preference
    # for the chosen answer compared with the reference model.
    chosen_log_ratios = (
        np.array(log_probs_chosen_policy)
        - np.array(log_probs_chosen_ref)
    )


    # Do the same thing for the rejected answers.
    #
    # Example:
    # [-3.0, -2.5] - [-2.5, -2.0]
    #
    # = [-3.0 - (-2.5), -2.5 - (-2.0)]
    #
    # = [-0.5, -0.5]
    #
    # The negative value means the policy gives the rejected answer
    # relatively less probability than the reference model.
    rejected_log_ratios = (
        np.array(log_probs_rejected_policy)
        - np.array(log_probs_rejected_ref)
    )


    # Now compare the chosen answer against the rejected answer.
    #
    # chosen_log_ratios   = [0.5, 0.5]
    # rejected_log_ratios = [-0.5, -0.5]
    #
    # Difference:
    #
    # [0.5 - (-0.5), 0.5 - (-0.5)]
    #
    # = [1.0, 1.0]
    #
    # Then multiply by beta = 0.5:
    #
    # [1.0, 1.0] * 0.5
    #
    # = [0.5, 0.5]
    #
    # These values are called logits.
    logits = beta * (chosen_log_ratios - rejected_log_ratios)


    # Convert each logit into a DPO loss.
    #
    # Our logits are [0.5, 0.5].
    #
    # sigmoid(0.5) ≈ 0.6225
    #
    # Therefore:
    #
    # -log(0.6225) ≈ 0.4741
    #
    # So the two individual losses are approximately:
    #
    # [0.4741, 0.4741]
    losses = -np.log(1 / (1 + np.exp(-logits)))


    # Calculate the implicit reward of the chosen responses.
    #
    # chosen_log_ratios = [0.5, 0.5]
    # beta = 0.5
    #
    # [0.5, 0.5] * 0.5
    #
    # = [0.25, 0.25]
    #
    # So the chosen answers receive positive rewards.
    chosen_rewards = beta * chosen_log_ratios


    # Calculate the implicit reward of the rejected responses.
    #
    # rejected_log_ratios = [-0.5, -0.5]
    # beta = 0.5
    #
    # [-0.5, -0.5] * 0.5
    #
    # = [-0.25, -0.25]
    #
    # So the rejected answers receive negative rewards.
    rejected_rewards = beta * rejected_log_ratios


    # Return the final results.
    #
    # np.mean(losses):
    # mean([0.4741, 0.4741]) = 0.4741
    #
    # chosen_rewards = [0.25, 0.25]
    # rejected_rewards = [-0.25, -0.25]
    #
    # round(..., 4) keeps everything to 4 decimal places.
    return {
        "loss": round(float(np.mean(losses)), 4),
        "chosen_rewards": [round(float(x), 4) for x in chosen_rewards],
        "rejected_rewards": [round(float(x), 4) for x in rejected_rewards]
    }