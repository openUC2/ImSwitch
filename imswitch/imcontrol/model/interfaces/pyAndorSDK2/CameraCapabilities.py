from imswitch.imcontrol.model.interfaces.pyAndorSDK2 import atmcd, atmcd_capabilities


class CapabilityHelper:
    """A helper class for extra device information from connected device

    This class provides several methods for extracting information about the current device.
    It

    Attributes
    ----------
        param1 : atmcd 
            An active atmcd/sdk object
    """

    def __init__(self, sdk: atmcd) -> None:
        (_, caps) = sdk.GetCapabilities()
        self.caps = caps
        self.sdk = sdk

    def print_all(self):
        """ returns a printout of each set of modes and functions
        """
        self.print_acquisition_modes()
        print("")
        self.print_read_modes()
        print("")
        self.print_trigger_modes()
        print("")
        self.print_camera_types()
        print("")
        self.print_pixel_modes()
        print("")
        self.print_set_functions()
        print("")
        self.print_get_functions()
        print("")
        self.print_features()
        print("")
        self.print_pci_card()
        print("")
        self.print_emgain_compatilibity()
        print("")
        self.print_FTRead_modes()
        print("")
        self.print_features2()
        print("")

    def print_acquisition_modes(self):
        """Prints available acquisition modes
        """
        mode = atmcd_capabilities.acquistionModes
        val = self.caps.ulAcqModes
        print("Available Acquisition modes")
        for i in range(0, len(bin(val))):
            self.__get_bit(val, i, mode)

    def print_read_modes(self):
        """Prints available read modes
        """
        mode = atmcd_capabilities.readmodes
        val = self.caps.ulReadModes
        print("Available Read modes")
        for i in range(0, len(bin(val))):
            self.__get_bit(val, i, mode)

    def print_trigger_modes(self):
        """Prints available pixel modes
        """
        mode = atmcd_capabilities.PixelModes
        val = self.caps.ulPixelMode

        print("Available Pixel Modes")
        for i in range(0, len(bin(val))):
            self.__get_bit(val, i, mode)

    def print_camera_types(self):
        """Prints camera type
        """
        mode = atmcd_capabilities.cameratype
        val = self.caps.ulCameraType

        print("Camera Type")
        for i in range(0, len(bin(val))):
            self.__get_bit(val, i, mode)

    def print_pixel_modes(self):
        """Prints available pixel modes
        """
        mode = atmcd_capabilities.PixelModes
        val = self.caps.ulPixelMode

        print("Available Pixel Modes")
        for i in range(0, len(bin(val))):
            self.__get_bit(val, i, mode)

    def print_set_functions(self):
        """Prints setter functions
        """
        mode = atmcd_capabilities.SetFunctions
        val = self.caps.ulSetFunctions

        print("Available Set Functions")
        for i in range(0, len(bin(val))):
            self.__get_bit(val, i, mode)

    def print_get_functions(self):
        """Prints available read modes
        """
        mode = atmcd_capabilities.GetFunctions
        val = self.caps.ulGetFunctions

        print("Available get Functions")
        for i in range(0, len(bin(val))):
            self.__get_bit(val, i, mode)

    def print_features(self):
        """Prints available features
        """
        mode = atmcd_capabilities.Features
        val = self.caps.ulFeatures

        print("Available Features")
        for i in range(0, len(bin(val))):
            self.__get_bit(val, i, mode)

    def print_pci_card(self):
        """Prints maximum speed of Pci card in Hz
        """
        val = self.caps.ulPCICard

        print("Pci max speed in Hz ")
        print("- {}".format(val))

    def print_step_modes(self):
        """Prints step type
        """
        mode = atmcd_capabilities.stepmodes
        val = self.caps.ulstepmodes

        print("Available Functions")
        for i in range(0, len(bin(val))):
            self.__get_bit(val, i, mode)

    def print_emgain_compatilibity(self):
        """Prints available Em gain modes
        """
        mode = atmcd_capabilities.EmGainModes
        val = self.caps.ulEMGainCapability

        print("Available Emgain compatibility Modes")
        for i in range(0, len(bin(val))):
            self.__get_bit(val, i, mode)

    def print_FTRead_modes(self):
        """Prints available Frame Transfer compatible Read modes
        """
        mode = atmcd_capabilities.readmodes
        val = self.caps.ulFeatures2

        print("Available FT compatible Read Modes")
        for i in range(0, len(bin(val))):
            self.__get_bit(val, i, mode)

    def print_features2(self):
        """Prints available features
        """
        mode = atmcd_capabilities.Features
        val = self.caps.ulFeatures

        print("Available Features")
        for i in range(0, len(bin(val))):
            self.__get_bit(val, i, mode)

    def __get_bit(self, number, bitNumber, mode):
        """converts from a long to binary data  
        """
        mask = 1 << bitNumber
        bit = (number & mask) >> bitNumber
        if bit != 1:
            pass
        else:
            self.__iterate_through_enum(int(mask), mode)

    def __iterate_through_enum(self, num, mode):
        """Checks provided enum for compatible value
        """
        for x in mode:
            if x.value == num:
                temp = x.name.split('_', -1)
                print("- {},".format(temp[2]))
