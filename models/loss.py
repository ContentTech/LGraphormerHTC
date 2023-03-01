import torch


def multilabel_categorical_crossentropy(y_true, y_pred):
    loss_mask = y_true != -100
    y_true = y_true.masked_select(loss_mask).view(-1, y_pred.size(-1))
    y_pred = y_pred.masked_select(loss_mask).view(-1, y_true.size(-1))
    y_pred = (1 - 2 * y_true) * y_pred
    y_pred_neg = y_pred - y_true * 1e12
    y_pred_pos = y_pred - (1 - y_true) * 1e12
    zeros = torch.zeros_like(y_pred[:, :1])
    y_pred_neg = torch.cat([y_pred_neg, zeros], dim=-1)
    y_pred_pos = torch.cat([y_pred_pos, zeros], dim=-1)
    neg_loss = torch.logsumexp(y_pred_neg, dim=-1)
    pos_loss = torch.logsumexp(y_pred_pos, dim=-1)
    return (neg_loss + pos_loss).mean()



def multilabel_categorical_BCE_weighted(y_true, y_pred, weight):
    loss_mask = y_true != -100
    y_true = y_true.masked_select(loss_mask).view(-1, y_pred.size(-1))
    y_pred = y_pred.masked_select(loss_mask).view(-1, y_true.size(-1))
    # deal with the weight
    size = y_true.shape
    batch_size = size[0]
    weight_pos = []
    for i in range(batch_size):
        weight_pos.append(weight)
    alpha_weight = torch.tensor(weight_pos, device=y_true.device).float()
    # y_pred = alpha_weight * y_pred
    # compute the loss
    per_entry_cross_ent = -y_true * torch.log(
        torch.clamp(y_pred, 1e-8, 1.0)) - (1.0 - y_true) * torch.log(torch.clamp(1.0 - y_pred, 1e-8, 1.0))
    per_entry_cross_ent = alpha_weight*per_entry_cross_ent
    return per_entry_cross_ent.mean()






