import numpy as np
from astropy import units as au
from astropy.units import Quantity


def rolling_window(a, window, padding='same'):
    """
    Produces a rolling window view of array.

    Args:
        a: ndarray
            Array to produce a rolling view over. The rolling view is over the last axis.
        window: int
            Size of rolling window
        padding: str
            Type of padding, if 'same' then using reflecting padding so that rolling axis stays the same length.

    Returns: ndarray
        If `a` is shape [..., T] this returns an array of shape [..., T, window] if padding == 'same'
        otherwise a shape [..., T - window + 1, window]

    Examples:
        >>> a = np.arange(5) #0,1,2,3,4
        >>> print(rolling_window(a,3, padding='same'))
        [[1 0 1]
         [0 1 2]
         [1 2 3]
         [2 3 4]
         [3 4 3]]
        >>> print(rolling_window(a,3,padding=None))
        [[0 1 2]
         [1 2 3]
         [2 3 4]]
        # Using it to perform rolling mean
        >>> b = rolling_window(a,3,padding='same')
        >>> print(np.mean(b, axis=-1))
        [0.66666667 1.         2.         3.         3.33333333]

    """
    if padding.lower() == 'same':
        pad_start = np.zeros(len(a.shape), dtype=np.int32)
        pad_start[-1] = window // 2
        pad_end = np.zeros(len(a.shape), dtype=np.int32)
        pad_end[-1] = (window - 1) - pad_start[-1]
        pad = list(zip(pad_start, pad_end))
        a = np.pad(a, pad, mode='reflect')
    shape = a.shape[:-1] + (a.shape[-1] - window + 1, window)
    strides = a.strides + (a.strides[-1],)
    return np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)

def test_rolling_window():
    a = np.arange(5)
    print(rolling_window(a,3,padding='same'))
    print(rolling_window(a,3,padding='valid'))
    print(rolling_window(a, 3, padding='same').mean(-1))


def _validate_type(name, value, expected_type):
    if not isinstance(value, expected_type):
        raise TypeError("{name} should be {expected_type}, got {type}.".format(name=name, expected_type=expected_type,
                                                                               type=type(value)))


def _validate_unit_type(name, value, expected_unit):
    _validate_type(name, value, Quantity)
    if isinstance(expected_unit, Quantity):
        expected_unit = expected_unit.unit
    if au.get_physical_type(value.unit) != au.get_physical_type(expected_unit):
        raise ValueError("{name} units should be {expected_unit}, got {type}.".format(name=name,
                                                                                      expected_unit=au.get_physical_type(
                                                                                          expected_unit),
                                                                                      type=au.get_physical_type(
                                                                                          value.unit)))


