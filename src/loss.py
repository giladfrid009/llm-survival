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

def prop_loss(input, meta):
    """ Custom prop_loss function.

    Args:
        input ([torch.tensor]): model prediction for probability of event at any given time
        meta ([tuple]): tuple of tensors including prop, the proportion of events and total, the number of samples
        L1_reg (float): L1 regularization parameter, used on the input sigmoid output

    Returns:
        [torch.tensor]: model loss, negative log likelihood
    """
    return F.binary_cross_entropy_with_logits(input[:,1], meta[0].float(), weight=meta[1])
