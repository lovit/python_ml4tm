import torch
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


def train_batch(model, data_loader, loss_func, optimizer,
    epoch, epochs, num_batches, verbose_batch=50, new_line=False):
    """
    Arguments : [model, data_loader, loss_func, optimizer,
        epoch, epochs, num_batches, verbose_batch=50, new_line=False]

    Returns : model, loss_sum
        model : torch.nn.Module
        loss_sum : torch.FloatTensor
    """

    # (correct, incorrect)
    num_corrects = torch.zeros(2, dtype=torch.float)
    loss_sum = torch.FloatTensor([0])
    for i_batch, (x_batch, y_batch, len_batch) in enumerate(data_loader):

        # clean-up optimizer
        optimizer.zero_grad()

        # prepare batch process
        packed_in = pack_padded_sequence(x_batch, len_batch,
            batch_first=True, enforce_sorted=False)

        # sorted y batch
        _, _, sorted_indices, _ = packed_in
        sorted_y_batch = y_batch[sorted_indices]

        # define loss
        y_pred = model(packed_in)
        loss = loss_func(y_pred, sorted_y_batch)

        # back-propagation
        loss.backward()
        optimizer.step()

        # cumulate temporal loss
        loss_sum += loss.detach()

        # train accuracy
        _, y_pred_idx = torch.max(y_pred, dim=1)
        n_correct = (y_pred_idx == sorted_y_batch).sum()
        num_corrects[0] += n_correct
        num_corrects[1] += (len(len_batch) - n_correct)

        if i_batch % verbose_batch == 0:
            verbose(epoch, epochs, i_batch, num_batches, loss_sum, new_line)
    verbose(epoch, epochs, i_batch, num_batches, loss_sum, new_line)
    train_accuracy = num_corrects[0] / num_corrects.sum()
    return model, loss_sum, train_accuracy

def verbose(epoch, epochs, i_batch, num_batches, loss_sum, new_line=False):
    loss_avg = float(loss_sum.numpy() / (1 + i_batch))
    args = (epoch, epochs, i_batch, num_batches, loss_avg, '\n' if new_line else '')
    print('\r{} / {} epochs, {} / {} batches, loss_avg = {:.4}{}'.format(*args), end='')
