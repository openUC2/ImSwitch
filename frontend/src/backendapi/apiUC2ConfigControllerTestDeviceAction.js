import createAxiosInstance from "./createAxiosInstance";

/**
 * Send a hardware validation/test command to a freshly-flashed slave device
 * over serial. Supports motor, ledarray, and laser commands.
 *
 * @param {Object} opts
 * @param {string} opts.port - Serial port device path (e.g. "/dev/ttyACM0")
 * @param {"motor"|"ledarray"|"laser"} opts.deviceType - Test type
 * @param {number} [opts.baud=115200] - Serial baudrate (overridable)
 * @param {number} [opts.timeout=2.0] - Serial read timeout in seconds
 * @param {number} [opts.stepperid=1] - Motor: stepper id
 * @param {number} [opts.speed=2000] - Motor: speed
 * @param {number} [opts.position=1000] - Motor: target position (steps)
 * @param {number} [opts.isabs=0] - Motor: absolute (1) or relative (0)
 * @param {number} [opts.r=25] - LED: red 0..255
 * @param {number} [opts.g=25] - LED: green 0..255
 * @param {number} [opts.b=25] - LED: blue 0..255
 * @param {string} [opts.ledAction="fill"] - LED: action ("fill", "off", ...)
 * @param {number} [opts.laserid=1] - Laser: LASERid (0..4)
 * @param {number} [opts.laserval=118] - Laser: LASERval (0..1023)
 * @returns {Promise<Object>} Result with status, command, response
 */
const apiUC2ConfigControllerTestDeviceAction = async ({
  port,
  deviceType,
  baud = 115200,
  timeout = 2.0,
  stepperid = 1,
  speed = 2000,
  position = 1000,
  isabs = 0,
  r = 25,
  g = 25,
  b = 25,
  ledAction = "fill",
  laserid = 1,
  laserval = 118,
} = {}) => {
  const axiosInstance = createAxiosInstance();
  const response = await axiosInstance.post(
    "/UC2ConfigController/testDeviceAction",
    null,
    {
      params: {
        port,
        device_type: deviceType,
        baud,
        timeout,
        stepperid,
        speed,
        position,
        isabs,
        r,
        g,
        b,
        led_action: ledAction,
        laserid,
        laserval,
      },
      timeout: 15000,
    }
  );
  return response.data;
};

export default apiUC2ConfigControllerTestDeviceAction;
