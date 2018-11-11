import numpy as np


def seizure_indices(time_axis, starts, ends):
    """Get the seizure indices for the training set

    creates sz_idx, list of list of seizure indices, in order of train eegs
    """

    sz_idx = np.zeros_like(time_axis, dtype=bool)
    for start, end in zip(starts, ends):
        sz_idx += (time_axis >= start) * (time_axis < end)
    return sz_idx


def Xs_to_channel_data(Xs):
    """Concatenate data for all channels

    inputs:
        Xs - list of lists of feautures

    returns:
        data_by_channel - list of stacked data
    """
    data_by_channel = []
    nchns = len(Xs[0])
    for chn in range(nchns):
        data = np.vstack((X[chn] for X in Xs))
        data_by_channel.append(data)
    return data_by_channel


def concatenate_nus(nus):
    """Concatenate a list of posteriors

    inputs:
        nus - list of lists of posteriors

    returns:
        nus_by_channel: list of posteriors
    """
    nus_by_channel = []
    nchns = len(nus[0])
    for chn in range(nchns):
        nu = np.concatenate([nu[chn] for nu in nus])
        nus_by_channel.append(nu)
    return nus_by_channel


def concatenate_labels(Ys):
    """Concatenate seizure indices, or any other list of arrays
    """
    return np.concatenate([y for y in Ys])


def channel_list_arrays_to_image(X):
    """List of 1d arrays to 2d image

    inputs - X, we assume X is a list of arrays length T

    returns:
        T x nchns matrix
    """
    return np.vstack(X).T


def X_to_stacked_features(X):
    """List of 2d arrays to 2d image

    inputs - X, we assume X is a list of matrices (T * d)

    returns:
        T x (nchns * d) matrix
    """
    return np.hstack(X)


def Xs_to_stacked_features(Xs):
    """List of list of 2d arrays to 2d image

    inputs - X, we assume X is a list of lists of matrices

    returns:
        T x (nchns * d) matrix
    """
    x_stacked = []
    for X in Xs:
        x_stacked.append(X_to_stacked_features(X))
    return np.vstack(x_stacked)


def get_batches(n, batch_size):
    """ Create minibatch indices """
    # Get a random permutation
    perm = np.random.permutation(n)

    # Loop over number of batches and get indices
    nbatches = int(np.ceil(n / batch_size))
    indices = []
    for ii in range(nbatches):
        indices.append(perm[ii * batch_size: (ii + 1) * batch_size])
    return indices

# def eegs_to_channel_data(data):
#     """
#     Given a list of EEGs, separate the channels

#     inputs:
#         data - a list of EEGs

#     returns:
#         data_by_channel - a list of numpy matrices containing channel data
#         labels - numpy array of labels
#     """
#     data_by_channel = []
#     # Initialize the labels and data matrices
#     labels = np.asarray(data[0].sz_idx, dtype=int)
#     for chn in range(data[0].nnodes):
#         data_by_channel.append(data[0].Y[chn])
#     # Concatenate labels and stack feature vectors
#     for eeg in data[1:]:
#         labels = np.concatenate((labels, np.asarray(eeg.sz_idx, dtype=int)))
#         for chn in range(eeg.nnodes):
#             data_by_channel[chn] = np.vstack((data_by_channel[chn],
#                                               eeg.Y[chn]))
#     return data_by_channel, labels


# def Xs_to_concatenated_channel_feats(Xs, eeg_infos):
#     """Takes features, return stacked feature vectors and labels

#     inputs:
#         Xs - list of lists of features
#         eeg_infos - corresponding list of eeg_infos

#     returns:
#         concatenated_data - all channel data concatenated
#         labels - numpy array of labels
#     """
#     # Loop over the data for dimensions and initialize
#     nfeats = sum([data[0].nfeats[chn] for chn in range(data[0].nnodes)])
#     nwindows = sum([eeg.nwindows for eeg in data])
#     # Initialize the numpy arrays
#     concatenated_data = np.zeros([nwindows, nfeats])
#     labels = np.zeros(nwindows)
#     # Loop over all EEGs and copy data
#     row_idx = 0
#     for eeg in data:
#         col_idx = 0
#         labels[row_idx:row_idx + eeg.nwindows] = eeg.sz_idx
#         for chn in range(eeg.nnodes):
#             concatenated_data[row_idx:row_idx + eeg.nwindows,
#                               col_idx:col_idx + eeg.nfeats[chn]] = eeg.Y[chn]
#             col_idx += eeg.nfeats[chn]
#         row_idx += eeg.nwindows
#     return concatenated_data, labels


# def eeg_to_concatenated_channel_feats(eeg):
#     """
#     Given an EEG, return stacked feature vectors

#     inputs:
#         data - a list of EEGs

#     returns:
#         concatenated_data - all channel data concatenated
#     """
#     # Initialize
#     nfeats = sum([eeg.nfeats[chn] for chn in range(eeg.nnodes)])
#     X = np.zeros([eeg.nwindows, nfeats])
#     # Loop over channels and concatenate
#     col_idx = 0
#     for chn in range(eeg.nnodes):
#         X[:, col_idx:col_idx + eeg.nfeats[chn]] = eeg.Y[chn]
#         col_idx += eeg.nfeats[chn]
#     return X
