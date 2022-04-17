
def aggregate_pieces(x, mask, reduction='first'):
    import torch
    """
    :param x: tensor of shape (seq_len) or (seq_len, hdim)
    :param mask: bool tensor of shape (seq_len)
    :param reduction: aggregation strategy (first, max, sum, mean)

    :returns: <s> word_1 word_2 ... </s>
    where word_i = aggregate(piece_i_1, piece_i_2, ...)
    """
    # mark <s> and </s> as True
    special_mask = mask.clone()
    special_mask[0] = special_mask[-1] = True
    if reduction == 'first':
        return x[special_mask.bool()]
    elif reduction == 'sum' or reduction == 'mean' or reduction == 'max':
        idx = special_mask.long().cumsum(dim=-1) - 1
        idx_unique_count = torch.bincount(idx)
        num_unique = idx_unique_count.shape[-1]
        if reduction == 'sum':
            res = torch.zeros(num_unique, device=x.device, dtype=x.dtype).scatter_add(0, idx, x)
        elif reduction == 'mean':
            res = torch.zeros(num_unique, device=x.device, dtype=x.dtype).scatter_add(0, idx, x)
            res /= idx_unique_count.float()
        else:
            res = torch.stack([x[idx == i].max() for i in range(num_unique)]).to(x.device)
        return res.float()

    else:
        return x
