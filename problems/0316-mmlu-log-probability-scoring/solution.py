import numpy as np

def mmlu_log_prob_score(log_probs: list, correct_answers: list) -> dict:
    """
    Compute MMLU-style log-probability scoring metrics.
    
    Args:
        log_probs: List of lists, where each inner list contains 
                   log-probabilities for each answer choice
        correct_answers: List of correct answer indices (0-indexed)
    
    Returns:
        Dictionary with 'accuracy', 'predictions', and 'avg_correct_prob'
    """
    pred =[]
    predictions = []
    correct_prob = []
    for i in range(len(log_probs)):
        question_log_prob = np.array(log_probs[i])
        maxPos= int(np.argmax(question_log_prob))    
        predictions.append(maxPos)
        maxProb = question_log_prob[maxPos]
        question_log_prob = question_log_prob - maxProb
        prob = question_log_prob[maxPos]
        exp_probs = np.exp(question_log_prob)
        probabilities = exp_probs/ np.sum(exp_probs)
        correct_prob.append(probabilities[correct_answers[i]])

    accuracy = np.mean(np.array(predictions)==np.array(correct_answers))
    avg_correct_prob = np.mean(correct_prob)
    return {
    "accuracy": round(float(accuracy), 4),
    "predictions": predictions,
    "avg_correct_prob": round(float(avg_correct_prob), 4)
}