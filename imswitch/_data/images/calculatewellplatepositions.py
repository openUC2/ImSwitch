#%% compute the positions of wells in a wellplate

import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os

#%%
# read in the json file with the coordinates
pixelsize_eff = .31 # um from camera
overlap = 0.75 # 25% overlap
n_pix_x, n_pix_y = 4000,3000

# Get the directory of the current script
base_dir = os.path.dirname(os.path.abspath(__file__))

# Define the well plate JSON files using relative paths based on the script's location
mFiles = (
    os.path.join(base_dir, "6_well_plate.json"),
    os.path.join(base_dir, "24_well_plate.json"),
    os.path.join(base_dir, "96_well_plate.json"),
    os.path.join(base_dir, "4_slide_carrier.json"),
)
for mFile in mFiles:
    with open(mFile) as f:
        data = json.load(f)

    #%
    # iterate over all wells and compute the positions
    well_positions = []

    radius = data['ScanParameters']['well_radius'] # mm
    fov_physical_x = pixelsize_eff*n_pix_x*(overlap)/1e3
    fov_physical_y = pixelsize_eff*n_pix_y*(overlap)/1e3
    # compute positions of radius
    n_tiles_x = int(2*radius/fov_physical_x) # number of pixels in the radius
    n_tiles_y = int(2*radius/fov_physical_y) # number of pixels in the radius

    # % create xx/yy meshgrid
    xx,yy = np.meshgrid(fov_physical_x*np.arange(-n_tiles_x//2,n_tiles_x//2)+1,fov_physical_y*np.arange(-n_tiles_y//2,n_tiles_y//2)+1)
    circle = ((xx)**2+(yy)**2) < radius**2

    well_scan_locations = (xx[circle].flatten(),yy[circle].flatten())

    # display image with effective pixel sizes
    if 0:
        plt.imshow(circle)
        plt.show()
        plt.plot(well_scan_locations[0],well_scan_locations[1],'r.')
        plt.show()

    for well in data['ScanParameters']['wells']:
        center_x, center_y = well['positionX'], well['positionY']

        if 0:
            plt.plot(well_scan_locations[0]+center_x,well_scan_locations[1]+center_y,'r.')
            plt.plot(center_x,center_y,'b.')
        else:
            for x, y in zip(well_scan_locations[0]+center_x, well_scan_locations[1]+center_y):
                rect = patches.Rectangle((x - 3.4 / 2, y - 2.2 / 2), 3.4, 2.2, linewidth=1, edgecolor='r', facecolor='none')
                plt.gca().add_patch(rect)
    plt.show()





# %%

# %%

# %%

# %%

# %%

# %%

# %%

    # %%
