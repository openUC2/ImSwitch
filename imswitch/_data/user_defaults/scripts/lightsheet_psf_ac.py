import numpy as np
import time
import threading
import tifffile as tif
import os

mainWindow.setCurrentModule('imcontrol')

# Get device names
positionerName = api.imcontrol.getPositionerNames()[0]
laserName = api.imcontrol.getLaserNames()[0]


# Create directory to save images
save_dir = r"C:\Users\lihai\OneDrive\Desktop\study\images"
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

# Move to the starting position 3500 (absolute position move, blocking)
start_X = -4300
api.imcontrol.movePositioner(positionerName, "X", start_X, True, True)

# Prepare to capture images
mImageList = []

# Define movement parameters
step_size = 5
end_X = -3900
total_distance = end_X - start_X  # Calculate total movement distance
num_steps = int(total_distance / step_size)  # Calculate number of steps

# Start capturing images
for i in range(num_steps):
    # Relative move, step size 5
    api.imcontrol.movePositioner(positionerName, "X", step_size, False, True)

    # Wait 50ms before proceeding
    time.sleep(1)

    # Capture image
    image = api.imcontrol.snapImage(True, False)
    mImageList.append(image)

    # Optional: Print current position
    current_position = start_X + (i + 1) * step_size
    print(f"Image captured at position X = {current_position}.")

# Save image stack
stack_filename = os.path.join(save_dir, "imagepsf3.tif")  # Updated filename
tif.imwrite(stack_filename, mImageList)
print(f"Image stack saved: {stack_filename}")

# Return to starting position (relative move, negative value of total distance)
api.imcontrol.movePositioner(positionerName, "X", -total_distance, False, True)

print("Image capture completed.")
