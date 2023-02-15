import numpy as np


# =============================================================================
# encoding function: integer_encoding / one_hot_encoding / label_encoding
# =============================================================================
def integer_encoding(sequence):
    """
    Note: This function converts DNA sequence data to label sequence.
    Reference: https://elferachid.medium.com/one-hot-encoding-dna-92a1c29ba15a
        Input: DNA sequence data (String)
        Output: Label sequence (List)

    """

    mapping = {"A": 0, "C": 1, "G": 2, "T": 3, "N": 4}
    seq2 = [mapping[i] for i in sequence]

    return seq2


def one_hot_encoding(sequence):
    """
    Note: This function converts DNA sequence data [string] to one-hot sequence [torch.Tensor].
    Reference: https://elferachid.medium.com/one-hot-encoding-dna-92a1c29ba15a

    """

    mapping = {"A": 0, "C": 1, "G": 2, "T": 3, "N": 4}
    seq2 = [mapping[i] for i in sequence]

    return np.eye(5)[seq2]


def label_encoding(label):
    if label == 0:
        return [0, 1] # = non-viral genome
    else:
        return [1, 0] # = viral genome
