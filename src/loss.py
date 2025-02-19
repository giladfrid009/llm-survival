import torch
import torch.nn.functional as F

def survival_loss(input, meta):
    """Custom survival_loss function.

    Args:
        input ([torch.tensor]): model prediction for probability of survival event at any given time
        meta ([tuple]): tuple of tensors including t_tilde, the censored time and delta, the event indicator

    Returns:
        [torch.tensor]: model loss, negative log likelihood
    """
    t_tilde, delta = meta

    # Computr the loss using binary cross entropy
    prop_success = delta / t_tilde
    return F.binary_cross_entropy_with_logits(input[:,1], prop_success, weight=t_tilde)

    # # Make delta into int tensor
    # delta = delta.int()
        
    # eps = 1e-8  # small epsilon for numerical stability

    # # Predicted probability of event (ensure numerical stability)
    # p = input[:,1].clamp(eps, 1 - eps)

    # # Calculate log probabilities
    # log_p = torch.log(p)
    # log_1mp = torch.log(1 - p)
    
    # # Compute the negative log-likelihood:
    # # If event occurred (delta=1): loss = -[log(p) + (t_tilde-1)*log(1-p)]
    # # If censored (delta=0): loss = -[t_tilde*log(1-p)]
    # # Unified expression:
    # nll = - (delta * log_p + (t_tilde - delta) * log_1mp)

    # # Average over the batch
    # return nll.mean()

def prop_loss(input, meta):
    """ Custom prop_loss function.

    Args:
        input ([torch.tensor]): model prediction for probability of event at any given time
        meta ([tuple]): tuple of tensors including prop, the proportion of events and total, the number of samples

    Returns:
        [torch.tensor]: model loss, negative log likelihood
    """

    return F.binary_cross_entropy_with_logits(input[:,1], meta[0].float(), weight=meta[1])

    # prop, total = meta

    # eps = 1e-8  # small epsilon for numerical stability

    # # Predicted probability of event (ensure numerical stability)
    # p = input[:,1].clamp(eps, 1 - eps)

    # # Calculate log probabilities
    # log_p = torch.log(p)
    # log_1mp = torch.log(1 - p)

    # # Compute the negative log-likelihood:
    # # If event occurred: loss = -[log(p)]
    # # If censored: loss = -[log(1-p)]
    # nll = - (prop * log_p + (1 - prop) * log_1mp)

    # # Average over the batch
    # return nll.mean()
