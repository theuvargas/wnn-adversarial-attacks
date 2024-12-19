import numpy as np
from numba import jit
import ctypes as c
from scipy.stats import norm


# Converts a vector of booleans to an unsigned integer
#  i.e. (2**0 * xv[0]) + (2**1 * xv[1]) + ... + (2**n * xv[n])
# Inputs:
#  xv: The boolean vector to be converted
# Returns: The unsigned integer representation of xv
@jit(nopython=True, inline="always")
def input_to_value(xv):
    result = 0
    for i in range(xv.size):
        result += xv[i] << i
    return result


# Generates a matrix of random values for use as m-arrays for H3 hash functions
def generate_h3_values(num_inputs, num_entries, num_hashes):
    assert np.log2(num_entries).is_integer()
    shape = (num_hashes, num_inputs)
    values = np.random.randint(0, num_entries, shape)
    return values


# Implementes a single discriminator in the WiSARD model
# A discriminator is a collection of boolean LUTs with associated input sets
# During inference, the outputs of all LUTs are summed to produce a response
class Discriminator:
    # Constructor
    # Inputs:
    #  num_inputs:    The total number of inputs to the discriminator
    #  unit_inputs:   The number of boolean inputs to each LUT/filter in the discriminator
    #  unit_entries:  The size of the underlying storage arrays for the filters. Must be a power of two.
    #  unit_hashes:   The number of hash functions for each filter.
    #  random_values: If provided, is used to set the random hash seeds for all filters. Otherwise, each filter generates its own seeds.
    def __init__(
        self, num_inputs, unit_inputs, unit_entries, unit_hashes, random_values=None
    ):
        assert (num_inputs / unit_inputs).is_integer()
        self.num_filters = num_inputs // unit_inputs
        self.filters = [
            BloomFilter(unit_inputs, unit_entries, unit_hashes, random_values)
            for i in range(self.num_filters)
        ]

    # Performs a training step (updating filter values)
    # Inputs:
    #  xv: A vector of boolean values representing the input sample
    def train(self, xv):
        filter_inputs = xv.reshape(
            self.num_filters, -1
        )  # Divide the inputs between the filters
        for idx, inp in enumerate(filter_inputs):
            self.filters[idx].add_member(inp)

    # Performs an inference to generate a response (number of filters which return True)
    # Inputs:
    #  xv: A vector of boolean values representing the input sample
    # Returns: The response of the discriminator to the input
    def predict(self, xv):
        filter_inputs = xv.reshape(
            self.num_filters, -1
        )  # Divide the inputs between the filters
        response = 0
        for idx, inp in enumerate(filter_inputs):
            response += int(self.filters[idx].check_membership(inp))
        return response

    # Sets the bleaching value for all filters
    # See the BloomFilter implementation for more information on what this means
    # Inputs:
    #  bleach: The new bleaching value to set
    def set_bleaching(self, bleach):
        for f in self.filters:
            f.set_bleaching(bleach)

    # Binarizes all filters; this process is irreversible
    # See the BloomFilter implementation for more information on what this means
    def binarize(self):
        for f in self.filters:
            f.binarize()


# Top-level class for the WiSARD weightless neural network model
class WiSARD:
    # Constructor
    # Inputs:
    #  num_inputs:       The total number of inputs to the model
    #  num_classes:      The number of distinct possible outputs of the model; the number of classes in the dataset
    #  unit_inputs:      The number of boolean inputs to each LUT/filter in the model
    #  unit_entries:     The size of the underlying storage arrays for the filters. Must be a power of two.
    #  unit_hashes:      The number of hash functions for each filter.
    def __init__(self, num_inputs, num_classes, unit_inputs, unit_entries, unit_hashes):
        self.pad_zeros = (
            ((num_inputs // unit_inputs) * unit_inputs) - num_inputs
        ) % unit_inputs
        pad_inputs = num_inputs + self.pad_zeros
        self.input_order = np.arange(pad_inputs)  # Use each input exactly once
        np.random.shuffle(self.input_order)  # Randomize the ordering of the inputs
        random_values = generate_h3_values(unit_inputs, unit_entries, unit_hashes)
        self.discriminators = [
            Discriminator(
                self.input_order.size,
                unit_inputs,
                unit_entries,
                unit_hashes,
                random_values,
            )
            for i in range(num_classes)
        ]

    # Performs a training step (updating filter values) for all discriminators
    # Inputs:
    #  xv: A vector of boolean values representing the input sample
    def train(self, xv, label):
        xv = np.pad(xv, (0, self.pad_zeros))[self.input_order]  # Reorder input
        self.discriminators[label].train(xv)

    # Performs an inference with the provided input
    # Passes the input through all discriminators, and returns the one or more with the maximal response
    # Inputs:
    #  xv: A vector of boolean values representing the input sample
    # Returns: A vector containing the indices of the discriminators with maximal response
    def predict(self, xv):
        xv = np.pad(xv, (0, self.pad_zeros))[self.input_order]  # Reorder input
        responses = np.array([d.predict(xv) for d in self.discriminators], dtype=int)
        max_response = responses.max()
        return np.where(responses == max_response)[0]

    # Sets the bleaching value for all filters
    # See the BloomFilter implementation for more information on what this means
    # Inputs:
    #  bleach: The new bleaching value to set
    def set_bleaching(self, bleach):
        for d in self.discriminators:
            d.set_bleaching(bleach)

    # Binarizes all filters; this process is irreversible
    # See the BloomFilter implementation for more information on what this means
    def binarize(self):
        for d in self.discriminators:
            d.binarize()


### BLOOM FILTER ###


# Computes hash functions within the H3 family of integer-integer hashing functions,
#  as described by Carter and Wegman in the paper "Universal Classes of Hash Functions"
# This function requires more unique parameters than the Dietzfelbinger multiply-shift hash function, but avoids arithmetic
# Inputs:
#  xv: A bitvector to be hashed to an integer
#  m: An array of arrays of length equivalent to the length of xv, with entries of size equivalent to the hash size
@jit(nopython=True)
def h3_hash(xv, m):
    # selected_entries = np.where(xv, m, 0)
    selected_entries = xv * m  # np.where is unsupported in Numba
    # reduction_result = np.bitwise_xor.reduce(selected_entries, axis=1)
    reduction_result = np.zeros(
        m.shape[0], dtype=np.int64
    )  # ".reduce" is unsupported in Numba
    for i in range(m.shape[1]):
        reduction_result ^= selected_entries[:, i]
    return reduction_result


# Implements a Bloom filter, a data structure for approximate set membership
# A Bloom filter can return one of two results: "possibly a member", and "definitely not a member"
# The risk of false positives increases with the number of elements stored relative to the underlying array size
# This implementation generalizes the basic concept to incorporate the notion of bleaching from WNN research
# With bleaching, we replace seen/not seen bits in the data structure with counters
# Elements can now be added to the data structure multiple times
# Our results now become "possibly added at least <b> times" and "definitely added fewer than <b> times"
# Increasing the bleaching threshold (the value of b) can improve accuracy
# Once the final bleaching threshold has been selected, this can be converted to a traditional Bloom filter
#  by evaluating the predicate "d[i] >= b" for all entries in the filter's data array
class BloomFilter:
    # Constructor
    # Inputs:
    #  num_inputs:     The bit width of the input to the filter (assumes the underlying inputs are single bits)
    #  num_entries:    The size of the underlying array for the filter. Must be a power of two. Increasing this reduces the risk of false positives.
    #  num_hashes:     The number of hash functions for the Bloom filter. This has a complex relation with false-positive rates
    #  hash_constants: Constant parameters for H3 hash
    def __init__(self, num_inputs, num_entries, num_hashes, hash_constants):
        self.num_inputs, self.num_entries, self.num_hashes = (
            num_inputs,
            num_entries,
            num_hashes,
        )
        self.hash_values = hash_constants
        self.index_bits = int(np.log2(num_entries))
        self.data = np.zeros(num_entries, dtype=int)
        self.bleach = np.array(1, dtype=int)

    # Implementation of the check_membership function
    # Coding in this style (as a static method) is necessary to use Numba for JIT compilation
    @staticmethod
    @jit(nopython=True)
    def __check_membership(xv, hash_values, bleach, data):
        # hash_results = dietzfelbinger_hash(x, a_values, b_values, num_inputs, index_bits)
        hash_results = h3_hash(xv, hash_values)
        least_entry = data[
            hash_results
        ].min()  # The most times the entry has possibly been previously seen
        return least_entry >= bleach

    # Check whether a value is a member of this filter (i.e. possibly seen at least b times)
    # Inputs:
    #  xv:              The bitvector to check the membership of
    # Returns: A boolean, which is true if xv has possibly been seen at least b times, and false if it definitely has not been
    def check_membership(self, xv):
        return BloomFilter.__check_membership(
            xv, self.hash_values, self.bleach, self.data
        )

    # Implementation of the add_member function
    # Coding in this style (as a static method) is necessary to use Numba for JIT compilation
    @staticmethod
    @jit(nopython=True)
    def __add_member(xv, hash_values, data):
        hash_results = h3_hash(xv, hash_values)
        least_entry = data[
            hash_results
        ].min()  # The most times the entry has possibly been previously seen
        data[hash_results] = np.maximum(
            data[hash_results], least_entry + 1
        )  # Increment upper bound

    # Register a bitvector / increment its encountered count in the filter
    # Inputs:
    #  xv: The bitvector
    def add_member(self, xv):
        BloomFilter.__add_member(xv, self.hash_values, self.data)

    # Set the bleaching threshold, which is used to exclude members which have not possibly been seen at least b times
    # Inputs:
    #  bleach: The new value for b
    def set_bleaching(self, bleach):
        self.bleach[...] = bleach

    # Converts the filter into a "canonical" Bloom filter, with all entries either 0 or 1 and bleaching of 1
    # This operation will not impact the result of the check_membership function for any input
    # This operation is irreversible, so shouldn't be done until the optimal bleaching value has been selected
    def binarize(self):
        self.data = (self.data >= self.bleach).astype(int)
        self.set_bleaching(1)


# Convert input dataset to binary representation
# Use a thermometer encoding with a configurable number of bits per input
# A thermometer encoding is a binary encoding in which subsequent bits are set as the value increases
#  e.g. 0000 => 0001 => 0011 => 0111 => 1111
def binarize_datasets(
    train_dataset,
    test_dataset,
    bits_per_input,
    separate_validation_dset=None,
    train_val_split_ratio=0.9,
):
    # Given a Gaussian with mean=0 and std=1, choose values which divide the distribution into regions of equal probability
    # This will be used to determine thresholds for the thermometer encoding
    std_skews = [
        norm.ppf((i + 1) / (bits_per_input + 1)) for i in range(bits_per_input)
    ]

    # print("Binarizing train/validation dataset")
    train_inputs = []
    train_labels = []
    for d in train_dataset:
        # Expects inputs to be already flattened numpy arrays
        train_inputs.append(d[0])
        train_labels.append(d[1])
    train_inputs = np.array(train_inputs)
    train_labels = np.array(train_labels)
    use_gaussian_encoding = True
    if use_gaussian_encoding:
        mean_inputs = train_inputs.mean(axis=0)
        std_inputs = train_inputs.std(axis=0)
        train_binarizations = []
        for i in std_skews:
            train_binarizations.append(
                (train_inputs >= mean_inputs + (i * std_inputs)).astype(c.c_ubyte)
            )
    else:
        min_inputs = train_inputs.min(axis=0)
        max_inputs = train_inputs.max(axis=0)
        train_binarizations = []
        for i in range(bits_per_input):
            train_binarizations.append(
                (
                    train_inputs
                    > min_inputs
                    + (((i + 1) / (bits_per_input + 1)) * (max_inputs - min_inputs))
                ).astype(c.c_ubyte)
            )

    # Creates thermometer encoding
    train_inputs = np.concatenate(train_binarizations, axis=1)

    # Ideally, we would perform bleaching using a separate dataset from the training set
    #  (we call this the "validation set", though this is arguably a misnomer),
    #  since much of the point of bleaching is to improve generalization to new data.
    # However, some of the datasets we use are simply too small for this to be effective;
    #  a very small bleaching/validation set essentially fits to random noise,
    #  and making the set larger decreases the size of the training set too much.
    # In these cases, we use the same dataset for training and validation
    if separate_validation_dset is None:
        separate_validation_dset = len(train_inputs) > 10000
    if separate_validation_dset:
        split = int(train_val_split_ratio * len(train_inputs))
        val_inputs = train_inputs[split:]
        val_labels = train_labels[split:]
        train_inputs = train_inputs[:split]
        train_labels = train_labels[:split]
    else:
        val_inputs = train_inputs
        val_labels = train_labels

    # print("Binarizing test dataset")
    test_inputs = []
    test_labels = []
    for d in test_dataset:
        # Expects inputs to be already flattened numpy arrays
        test_inputs.append(d[0])
        test_labels.append(d[1])
    test_inputs = np.array(test_inputs)
    test_labels = np.array(test_labels)
    test_binarizations = []
    if use_gaussian_encoding:
        for i in std_skews:
            test_binarizations.append(
                (test_inputs >= mean_inputs + (i * std_inputs)).astype(c.c_ubyte)
            )
    else:
        for i in range(bits_per_input):
            test_binarizations.append(
                (
                    test_inputs
                    > min_inputs
                    + (((i + 1) / (bits_per_input + 1)) * (max_inputs - min_inputs))
                ).astype(c.c_ubyte)
            )
    test_inputs = np.concatenate(test_binarizations, axis=1)

    return train_inputs, train_labels, val_inputs, val_labels, test_inputs, test_labels
