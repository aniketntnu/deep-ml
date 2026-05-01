import numpy as np
from collections import Counter

def kl_divergence_filter(reference_freq: dict, documents: list, threshold: float) -> dict:
    """Filter documents by KL divergence from a reference token distribution."""
    
    alpha = 1.0  # Laplace smoothing
    kl_values = []
    flagged = []
    kept = []
    
    for doc in documents:
        
        # Step 1: Count tokens in document
        doc_counts = Counter(doc)
        
        # Step 2: Build vocabulary (tokens in doc)
        vocab = list(doc_counts.keys())
        
        # Step 3: Build P_doc (document distribution)
        total_doc = sum(doc_counts.values())
        P = np.array([doc_counts[t] / total_doc for t in vocab])
        
        # Step 4: Build P_ref with smoothing
        ref_counts = np.array([reference_freq.get(t, 0) for t in vocab])
        
        # Apply Laplace smoothing
        ref_counts = ref_counts + alpha
        
        total_ref = np.sum(ref_counts)
        Q = ref_counts / total_ref
        
        # Step 5: Compute KL divergence
        # KL(P || Q) = sum P * log(P / Q)
        kl = np.sum(P * np.log(P / Q))
        
        kl = round(float(kl), 4)
        kl_values.append(kl)
        
        # Step 6: Flag
        is_flagged = kl > threshold
        flagged.append(is_flagged)
        
        # Step 7: Keep if not flagged
        if not is_flagged:
            kept.append(doc)
    
    return {
        'kl_divergences': kl_values,
        'flagged': flagged,
        'kept': kept
    }