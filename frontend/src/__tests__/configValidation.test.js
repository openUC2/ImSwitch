/**
 * Test for UC2ConfigurationController configuration improvements
 */
import {
  validateConfiguration,
  validateJsonString,
  createConfigurationPreview,
} from "../utils/configValidation";

describe("Configuration Validation Utilities", () => {
  test("validateJsonString should handle valid JSON", () => {
    const validJson = '{"test": "value"}';
    const result = validateJsonString(validJson);

    expect(result.isValid).toBe(true);
    expect(result.error).toBe(null);
    expect(result.parsed).toEqual({ test: "value" });
  });

  test("validateJsonString should handle invalid JSON", () => {
    const invalidJson = '{"test": value}'; // missing quotes around value
    const result = validateJsonString(invalidJson);

    expect(result.isValid).toBe(false);
    expect(result.error).toBeTruthy();
    expect(result.parsed).toBe(null);
  });

  test("validateConfiguration should validate basic structure", () => {
    const validConfig = {
      detector: { camera1: { managerName: "CameraManager" } },
      actuators: { laser1: { managerName: "UC2ConfigLaser" } },
      positioners: { stage1: { managerName: "StageManager" } },
    };

    const result = validateConfiguration(validConfig);

    expect(result.isValid).toBe(true);
    expect(result.errors).toHaveLength(0);
  });

  test("validateConfiguration should detect missing structure", () => {
    const invalidConfig = {
      actuators: "not_an_object",
    };

    const result = validateConfiguration(invalidConfig);

    expect(result.isValid).toBe(false);
    expect(result.errors).toContain(
      "Actuators configuration must be an object"
    );
  });

  test("createConfigurationPreview should count devices correctly", () => {
    const config = {
      detector: { camera1: {} },
      actuators: {
        laser1: { managerName: "UC2ConfigLaser" },
        laser2: { managerName: "UC2ConfigLaser" },
        scanner1: { managerName: "UC2ConfigScanner" },
      },
      positioners: { stage1: {}, stage2: {} },
    };

    const preview = createConfigurationPreview(config);

    expect(preview.valid).toBe(true);
    expect(preview.detectors).toBe(1);
    expect(preview.actuators).toBe(3);
    expect(preview.positioners).toBe(2);
    expect(preview.lasers).toBe(2);
    expect(preview.scanners).toBe(1);
  });
});
