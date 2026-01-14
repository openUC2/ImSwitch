/**
 * Configuration validation utilities for UC2ConfigurationController
 */

/**
 * Validates a filename for cross-platform compatibility
 * @param {string} filename - The filename to validate (without .json extension)
 * @returns {object} - { isValid: boolean, error: string }
 */
export const validateFileName = (filename) => {
  // Check for empty or whitespace-only
  if (!filename || !filename.trim()) {
    return { isValid: false, error: "Filename cannot be empty" };
  }

  const trimmed = filename.trim();

  // Check length (255 is typical max, but we'll be conservative)
  // Subtract 5 for ".json" extension
  if (trimmed.length > 250) {
    return {
      isValid: false,
      error: "Filename is too long (max 250 characters)",
    };
  }

  // Invalid characters for Windows/Linux/macOS
  // eslint-disable-next-line no-control-regex
  const invalidChars = /[<>:"/\\|?*\x00-\x1F]/;
  if (invalidChars.test(trimmed)) {
    return {
      isValid: false,
      error: 'Filename contains invalid characters (< > : " / \\ | ? *)',
    };
  }

  // Reserved names on Windows (case-insensitive)
  const reservedNames = [
    "CON",
    "PRN",
    "AUX",
    "NUL",
    "COM1",
    "COM2",
    "COM3",
    "COM4",
    "COM5",
    "COM6",
    "COM7",
    "COM8",
    "COM9",
    "LPT1",
    "LPT2",
    "LPT3",
    "LPT4",
    "LPT5",
    "LPT6",
    "LPT7",
    "LPT8",
    "LPT9",
  ];

  const nameWithoutExtension = trimmed.replace(/\.json$/i, "");
  if (reservedNames.includes(nameWithoutExtension.toUpperCase())) {
    return {
      isValid: false,
      error: `"${nameWithoutExtension}" is a reserved system filename`,
    };
  }

  // Check for names ending with dot or space (not allowed on Windows)
  if (/[. ]$/.test(trimmed)) {
    return {
      isValid: false,
      error: "Filename cannot end with a dot or space",
    };
  }

  return { isValid: true, error: null };
};

/**
 * Validates JSON configuration structure and content
 * @param {object} config - The configuration object to validate
 * @returns {object} - { isValid: boolean, errors: array, warnings: array }
 */
export const validateConfiguration = (config) => {
  const errors = [];
  const warnings = [];

  if (!config || typeof config !== "object") {
    errors.push("Configuration must be a valid object");
    return { isValid: false, errors, warnings };
  }

  // Check for required basic structure (using actual config structure)
  if (!config.detectors && !config.detector) {
    warnings.push("No detector configuration found");
  }

  if (!config.lasers && !config.actuators) {
    warnings.push("No laser/actuator configuration found");
  }

  if (!config.positioners) {
    warnings.push("No positioners configuration found");
  }

  // Validate device configurations if present
  if (config.detectors && typeof config.detectors !== "object") {
    errors.push("Detectors configuration must be an object");
  }
  if (config.detector && typeof config.detector !== "object") {
    errors.push("Detector configuration must be an object");
  }

  if (config.lasers && typeof config.lasers !== "object") {
    errors.push("Lasers configuration must be an object");
  }
  if (config.actuators && typeof config.actuators !== "object") {
    errors.push("Actuators configuration must be an object");
  }

  if (config.positioners && typeof config.positioners !== "object") {
    errors.push("Positioners configuration must be an object");
  }

  // Check for common configuration issues in detectors
  if (config.detectors) {
    Object.entries(config.detectors).forEach(([key, detector]) => {
      if (!detector.managerName) {
        warnings.push(`Detector '${key}' missing managerName`);
      }
      if (!detector.managerProperties) {
        warnings.push(`Detector '${key}' missing managerProperties`);
      }
    });
  }

  // Check for common configuration issues in lasers
  if (config.lasers) {
    Object.entries(config.lasers).forEach(([key, laser]) => {
      if (!laser.managerName) {
        warnings.push(`Laser '${key}' missing managerName`);
      }
      if (!laser.managerProperties) {
        warnings.push(`Laser '${key}' missing managerProperties`);
      }
    });
  }

  // Check for common configuration issues in actuators (legacy support)
  if (config.actuators) {
    Object.entries(config.actuators).forEach(([key, actuator]) => {
      if (!actuator.managerName) {
        warnings.push(`Actuator '${key}' missing managerName`);
      }
      if (!actuator.managerProperties) {
        warnings.push(`Actuator '${key}' missing managerProperties`);
      }
    });
  }

  // Check for common configuration issues in positioners
  if (config.positioners) {
    Object.entries(config.positioners).forEach(([key, positioner]) => {
      if (!positioner.managerName) {
        warnings.push(`Positioner '${key}' missing managerName`);
      }
      if (!positioner.managerProperties) {
        warnings.push(`Positioner '${key}' missing managerProperties`);
      }
    });
  }

  const isValid = errors.length === 0;
  return { isValid, errors, warnings };
};

/**
 * Validates JSON string syntax
 * @param {string} jsonString - JSON string to validate
 * @returns {object} - { isValid: boolean, error: string|null, parsed: object|null }
 */
export const validateJsonString = (jsonString) => {
  try {
    const parsed = JSON.parse(jsonString);
    return { isValid: true, error: null, parsed };
  } catch (error) {
    return {
      isValid: false,
      error: error.message,
      parsed: null,
    };
  }
};

/**
 * Formats validation results for user display
 * @param {object} validationResult - Result from validateConfiguration
 * @returns {string} - Formatted message for display
 */
export const formatValidationMessage = (validationResult) => {
  const { isValid, errors, warnings } = validationResult;

  let message = "";

  if (isValid) {
    message = "✅ Configuration is valid";
    if (warnings.length > 0) {
      message += `\n⚠️ Warnings:\n${warnings.map((w) => `• ${w}`).join("\n")}`;
    }
  } else {
    message = "❌ Configuration has errors:";
    message += `\n${errors.map((e) => `• ${e}`).join("\n")}`;
    if (warnings.length > 0) {
      message += `\n⚠️ Warnings:\n${warnings.map((w) => `• ${w}`).join("\n")}`;
    }
  }

  return message;
};

/**
 * Creates a preview summary of the configuration
 * @param {object} config - Configuration object
 * @returns {object} - Summary object with counts and key information
 */
export const createConfigurationPreview = (config) => {
  if (!config || typeof config !== "object") {
    return {
      valid: false,
      summary: "Invalid configuration",
      detectorCount: 0,
      laserCount: 0,
      actuatorCount: 0,
      positionerCount: 0,
      totalDevices: 0,
      devices: [],
    };
  }

  const devices = [];

  // Extract detector information (support both detectors and detector)
  let detectorCount = 0;
  const detectors = config.detectors || config.detector;
  if (detectors && typeof detectors === "object") {
    detectorCount = Object.keys(detectors).length;
    Object.entries(detectors).forEach(([key, detector]) => {
      devices.push({
        name: key,
        type: "Detector",
        manager: detector.managerName || "Unknown",
      });
    });
  }

  // Extract laser information
  let laserCount = 0;
  if (config.lasers && typeof config.lasers === "object") {
    laserCount = Object.keys(config.lasers).length;
    Object.entries(config.lasers).forEach(([key, laser]) => {
      devices.push({
        name: key,
        type: "Laser",
        manager: laser.managerName || "Unknown",
      });
    });
  }

  // Extract actuator information (legacy support)
  let actuatorCount = 0;
  if (config.actuators && typeof config.actuators === "object") {
    actuatorCount = Object.keys(config.actuators).length;
    Object.entries(config.actuators).forEach(([key, actuator]) => {
      devices.push({
        name: key,
        type: "Actuator",
        manager: actuator.managerName || "Unknown",
      });
    });
  }

  // Extract positioner information
  let positionerCount = 0;
  if (config.positioners && typeof config.positioners === "object") {
    positionerCount = Object.keys(config.positioners).length;
    Object.entries(config.positioners).forEach(([key, positioner]) => {
      devices.push({
        name: key,
        type: "Positioner",
        manager: positioner.managerName || "Unknown",
      });
    });
  }

  return {
    valid: true,
    detectorCount,
    laserCount,
    actuatorCount,
    positionerCount,
    totalDevices: detectorCount + laserCount + actuatorCount + positionerCount,
    devices,
  };
};
