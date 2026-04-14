from enum import IntEnum, unique


@unique
class Read_Mode(IntEnum):
    """Read mode options
    """
    FULL_VERTICAL_BINNING = 0
    MULTI_TRACK = 1
    RANDOM_TRACK = 2
    SINGLE_TRACK = 3
    IMAGE = 4


@unique
class Trigger_Mode(IntEnum):
    """Trigger mode options
    """
    INTERNAL = 0
    EXTERNAL = 1
    EXTERNAL_START = 6
    EXTERNAL_EXPOSURE_BULB = 7
    EXTERNAL_FVB_EM = 9
    SOFTWARE_TRIGGER = 10
    EXTERNAL_CHARGE_SHIFTING = 12


@unique
class Acquisition_Mode(IntEnum):
    """Acquistion mode options
    """
    SINGLE_SCAN = 1
    ACCUMULATE = 2
    KINETICS = 3
    FAST_KINETICS = 4
    RUN_TILL_ABORT = 5


@unique
class Spool_Mode(IntEnum):
    """Spool mode options
    """
    FILE_32_BIT_SEQUENCE = 0
    # Format of data in files depends on whether multiple accumulations are being taken for each scan. Format will be 32-bit integer if data is being accumulated each scan; otherwise the format will be 16-bit integer.
    DATA_DEPENDENT_FORMAT = 1
    FILE_16_BIT_SEQUENCE = 2
    MULTIPLE_DIRECTORY_STRUCTURE = 3
    SPOOL_TO_RAM = 4
    SPOOL_TO_16_BIT_FITS = 5
    SPOOL_TO_SIF = 6
    SPOOL_TO_16_BIT_TIFF = 7
    COMPRESSED_MULTIPLE_DIRECTORY_STRUCTURE = 8


@unique
class Gate_Mode(IntEnum):
    """Gate mode options
    """
    FIRE_ANDED_WITH_THE_GATE_INPUT = 0
    GATING_CONTROLLED_FROM_FIRE_PULSE_ONLY = 1
    GATING_CONTROLLED_FROM_SMB_GATE_INPUT_ONLY = 2
    GATING_ON_CONTINUOUSLY = 3
    GATING_OFF_CONTINUOUSLY = 4
    GATE_USING_DDG = 5


@unique
class Shutter_Mode(IntEnum):
    """Shutter mode options
    """
    FULLY_AUTO = 0
    PERMANENTLY_OPEN = 1
    PERMANENTLY_CLOSED = 2
    OPEN_FOR_FVB_SERIES = 4
    OPEN_FOR_ANY_SERIES = 5
