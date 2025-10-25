# Stage Calibration API Documentation

This document describes the REST API endpoints for stage-to-camera affine calibration.

## Overview

The stage calibration API allows you to perform robust affine calibration between the microscope stage and camera through HTTP requests. This is useful for web-based interfaces and remote control.

## API Endpoints

All endpoints are accessible via the ImSwitch REST API. The base path depends on your ImSwitch configuration.

### 1. Calibrate Stage (POST)

Perform affine stage-to-camera calibration.

**Endpoint:** `/calibrateStageAffine`

**Method:** POST

**Parameters:**
- `objectiveId` (string, optional): Identifier for the objective being calibrated. Default: "default"
- `stepSizeUm` (float, optional): Step size in microns. Default: 100.0. Recommended: 50-200
- `pattern` (string, optional): Movement pattern - "cross" or "grid". Default: "cross"
- `nSteps` (integer, optional): Number of steps in each direction. Default: 4
- `validate` (boolean, optional): Whether to validate the calibration. Default: true

**Example Request:**
```json
{
  "objectiveId": "10x",
  "stepSizeUm": 150.0,
  "pattern": "cross",
  "nSteps": 4,
  "validate": true
}
```

**Example Response (Success):**
```json
{
  "success": true,
  "objectiveId": "10x",
  "metrics": {
    "rmse_um": 0.234,
    "rotation_deg": 1.15,
    "scale_x_um_per_pixel": 0.500,
    "scale_y_um_per_pixel": 0.500,
    "quality": "excellent",
    "mean_correlation": 0.85,
    "n_inliers": 9,
    "n_outliers": 0
  },
  "validation": {
    "is_valid": true,
    "message": "Calibration passed all validation checks"
  }
}
```

**Example Response (Error):**
```json
{
  "success": false,
  "error": "Calibration failed: ..."
}
```

**Notes:**
- This operation takes 30-60 seconds to complete
- The microscope stage will move during calibration
- Ensure a calibration sample (e.g., grid pattern) is in focus before calling

### 2. Get Calibrated Objectives (GET)

Get a list of all objectives that have been calibrated.

**Endpoint:** `/getCalibrationObjectives`

**Method:** GET

**Parameters:** None

**Example Response:**
```json
{
  "success": true,
  "objectives": ["10x", "20x", "40x", "default"]
}
```

### 3. Get Calibration Data (GET)

Retrieve calibration data for a specific objective.

**Endpoint:** `/getCalibrationData`

**Method:** GET

**Parameters:**
- `objectiveId` (string, optional): Identifier for the objective. Default: "default"

**Example Request:**
```
GET /getCalibrationData?objectiveId=10x
```

**Example Response:**
```json
{
  "success": true,
  "objectiveId": "10x",
  "affineMatrix": [
    [0.500, 0.010, 0.0],
    [-0.010, 0.500, 0.0]
  ],
  "metrics": {
    "rmse_um": 0.234,
    "rotation_deg": 1.15,
    "scale_x_um_per_pixel": 0.500,
    "scale_y_um_per_pixel": 0.500,
    "quality": "excellent",
    "mean_correlation": 0.85,
    "condition_number": 2.3,
    "n_inliers": 9,
    "n_outliers": 0
  }
}
```

**Affine Matrix Format:**
The `affineMatrix` is a 2×3 matrix:
```
[a11  a12  tx]
[a21  a22  ty]
```
Where:
- `a11, a12, a21, a22`: 2×2 rotation/scale/shear matrix
- `tx, ty`: translation vector in microns

To transform pixel coordinates to stage coordinates:
```
stage_x = pixel_x * a11 + pixel_y * a12 + tx
stage_y = pixel_x * a21 + pixel_y * a22 + ty
```

### 4. Delete Calibration (POST/DELETE)

Delete calibration data for a specific objective.

**Endpoint:** `/deleteCalibration`

**Method:** POST or DELETE

**Parameters:**
- `objectiveId` (string, required): Identifier for the objective to delete

**Example Request:**
```json
{
  "objectiveId": "10x"
}
```

**Example Response:**
```json
{
  "success": true,
  "objectiveId": "10x",
  "message": "Calibration deleted for '10x'"
}
```

## Usage Examples

### Python (using requests)

```python
import requests
import json

# Base URL for your ImSwitch instance
BASE_URL = "http://localhost:8001"

# 1. Calibrate a 10x objective
calibration_params = {
    "objectiveId": "10x",
    "stepSizeUm": 150.0,
    "pattern": "cross",
    "validate": True
}

response = requests.post(
    f"{BASE_URL}/calibrateStageAffine",
    json=calibration_params
)
result = response.json()
print(f"Calibration quality: {result['metrics']['quality']}")
print(f"RMSE: {result['metrics']['rmse_um']:.3f} µm")

# 2. Get list of calibrated objectives
response = requests.get(f"{BASE_URL}/getCalibrationObjectives")
objectives = response.json()
print(f"Calibrated objectives: {objectives['objectives']}")

# 3. Get calibration data for 10x objective
response = requests.get(
    f"{BASE_URL}/getCalibrationData",
    params={"objectiveId": "10x"}
)
data = response.json()
affine_matrix = data['affineMatrix']
print(f"Affine matrix:\n{affine_matrix}")

# 4. Delete calibration
response = requests.post(
    f"{BASE_URL}/deleteCalibration",
    json={"objectiveId": "10x"}
)
print(response.json()['message'])
```

### JavaScript (using fetch)

```javascript
const BASE_URL = "http://localhost:8001";

// 1. Calibrate stage
async function calibrateStage() {
  const response = await fetch(`${BASE_URL}/calibrateStageAffine`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      objectiveId: "10x",
      stepSizeUm: 150.0,
      pattern: "cross",
      validate: true
    })
  });
  
  const result = await response.json();
  console.log(`Quality: ${result.metrics.quality}`);
  console.log(`RMSE: ${result.metrics.rmse_um} µm`);
  return result;
}

// 2. Get calibrated objectives
async function getObjectives() {
  const response = await fetch(`${BASE_URL}/getCalibrationObjectives`);
  const data = await response.json();
  console.log("Calibrated objectives:", data.objectives);
  return data.objectives;
}

// 3. Get calibration data
async function getCalibrationData(objectiveId) {
  const response = await fetch(
    `${BASE_URL}/getCalibrationData?objectiveId=${objectiveId}`
  );
  const data = await response.json();
  return data.affineMatrix;
}

// 4. Delete calibration
async function deleteCalibration(objectiveId) {
  const response = await fetch(`${BASE_URL}/deleteCalibration`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ objectiveId })
  });
  const result = await response.json();
  console.log(result.message);
}
```

### cURL

```bash
# 1. Calibrate stage
curl -X POST http://localhost:8001/calibrateStageAffine \
  -H "Content-Type: application/json" \
  -d '{
    "objectiveId": "10x",
    "stepSizeUm": 150.0,
    "pattern": "cross",
    "validate": true
  }'

# 2. Get calibrated objectives
curl http://localhost:8001/getCalibrationObjectives

# 3. Get calibration data
curl "http://localhost:8001/getCalibrationData?objectiveId=10x"

# 4. Delete calibration
curl -X POST http://localhost:8001/deleteCalibration \
  -H "Content-Type: application/json" \
  -d '{"objectiveId": "10x"}'
```

## Quality Metrics

The calibration returns several quality metrics:

- **rmse_um**: Root mean square error in microns (lower is better)
- **rotation_deg**: Rotation angle between stage and camera axes in degrees
- **scale_x_um_per_pixel**: Microns per pixel in X direction
- **scale_y_um_per_pixel**: Microns per pixel in Y direction
- **quality**: Overall quality assessment ("excellent", "good", "acceptable", "poor")
- **mean_correlation**: Mean correlation value for image matching (0-1, higher is better)
- **condition_number**: Matrix conditioning (lower is better, <100 is good)
- **n_inliers**: Number of valid measurement points
- **n_outliers**: Number of rejected outlier points

### Quality Thresholds

- **Excellent**: RMSE < 1.0 µm, correlation > 0.5
- **Good**: RMSE < 2.0 µm, correlation > 0.3
- **Acceptable**: RMSE < 5.0 µm
- **Poor**: RMSE ≥ 5.0 µm (should recalibrate)

## Error Handling

All endpoints return a JSON response with a `success` field:
- `success: true` - Operation completed successfully
- `success: false` - Operation failed, check the `error` field for details

Common errors:
- `"Camera stage mapping module not available"` - Required module not installed
- `"No calibration found for objective 'X'"` - Objective not calibrated yet
- `"Calibration failed: ..."` - Calibration process encountered an error

## Best Practices

1. **Sample Preparation**: Use a structured calibration sample (grid pattern, dots, etc.) in focus
2. **Step Size**: Choose appropriate step size based on objective:
   - 4x: 200 µm
   - 10x: 150 µm
   - 20x: 75 µm
   - 40x: 40 µm
3. **Pattern Selection**: Use "cross" for faster calibration (30 sec), "grid" for higher precision (60 sec)
4. **Validation**: Always enable validation to check calibration quality
5. **Multiple Objectives**: Calibrate each objective separately for best accuracy
6. **Recalibration**: Recalibrate if quality is "poor" or after significant hardware changes

## Integration with Web UI

The API is designed to be easily integrated into web-based microscope control interfaces:

1. Add calibration button in your web UI
2. Call `/calibrateStageAffine` when button is clicked
3. Display progress indicator (calibration takes 30-60 seconds)
4. Show quality metrics after completion
5. Use `/getCalibrationObjectives` to populate objective selector
6. Use `/getCalibrationData` to display current calibration status

## Troubleshooting

**Problem**: Calibration fails with high RMSE
- **Solution**: Check sample focus, try different position, verify stage axes configuration

**Problem**: Low correlation values
- **Solution**: Improve illumination, ensure sample has sufficient features

**Problem**: Module not available error
- **Solution**: Install camera_stage_mapping module dependencies (scipy, scikit-image)

**Problem**: Stage doesn't move during calibration
- **Solution**: Verify stage is connected and functional, check stage axis configuration

## See Also

- Main documentation: `docs/affine_calibration_guide.md`
- Implementation details: `docs/implementation_summary.md`
- Code examples: `examples/affine_calibration_examples.py`
