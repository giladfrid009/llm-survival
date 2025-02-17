import torch

def survival_loss(input, meta):
    """Custom survival_loss function.

    Args:
        input ([torch.tensor]): model prediction for probability of survival event at any given time
        meta ([tuple]): tuple of tensors including t_tilde, the censored time and delta, the event indicator

    Returns:
        [torch.tensor]: model loss, negative log likelihood
    """
    t_tilde, delta = meta

    # Make delta into int tensor
    delta = delta.int()
        
    eps = 1e-8  # small epsilon for numerical stability

    # Predicted probability of event (ensure numerical stability)
    p = input[:,:1].clamp(eps, 1 - eps)

    # Calculate log probabilities
    log_p = torch.log(p)
    log_1mp = torch.log(1 - p)
    
    # Compute the negative log-likelihood:
    # If event occurred (delta=1): loss = -[log(p) + (t_tilde-1)*log(1-p)]
    # If censored (delta=0): loss = -[t_tilde*log(1-p)]
    # Unified expression:
    nll = - (delta * log_p + (t_tilde - delta) * log_1mp)

    # Average over the batch
    return nll.mean()

# def prop_loss(input, meta):
#     """ Custom prop_loss function.
