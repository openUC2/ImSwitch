import json
import os

def generate_wellplate_json(
    name,
    pixel_image_x,
    pixel_image_y,
    phys_dim_x,
    phys_dim_y,
    phys_offset_x,
    phys_offset_y,
    image_path,
    num_wells_x,
    num_wells_y,
    center_x,
    center_y,
    well_spacing_x,
    well_spacing_y,
    well_radius
):
    wells = []
    start_x = center_x - ((num_wells_x - 1) * well_spacing_x / 2)
    start_y = center_y - ((num_wells_y - 1) * well_spacing_y / 2)

    idN = 0
    for i in range(num_wells_x):
        for j in range(num_wells_y):
            well_x = start_x + i * well_spacing_x
            well_y = start_y + j * well_spacing_y
            wells.append({
                "wellID": f"{chr(65+j)}{i+1}",
                "positionX": well_x,
                "positionY": well_y,
                "positionXpx": well_x * pixel_image_x / phys_dim_x,
                "positionYpx": well_y * pixel_image_y / phys_dim_y,
                "idX": i,
                "idY": j,
                "idN": idN,
                "positionXmin": well_x - well_radius,
                "positionXmax": well_x + well_radius,
                "positionYmin": well_y - well_radius,
                "positionYmax": well_y + well_radius
            })
            idN += 1

    data = {
        "ScanParameters": {
            "name": name,
            "pixelImageX": pixel_image_x,
            "pixelImageY": pixel_image_y,
            "physDimX": phys_dim_x,
            "physDimY": phys_dim_y,
            "note_on_offset": "the offset is measured from the left lower corner",
            "physOffsetX": phys_offset_x,
            "physOffsetY": phys_offset_y,
            "imagePath": image_path,
            "well_radius": well_radius,
            "wells": wells
        }
    }

    return data

imageFilePaths = [
                  'images/samplelayouts/6wellplate_1509x1010.png',
                  'images/samplelayouts/12wellplate_1509x1010.png',
                  'images/samplelayouts/24wellplate_1509x1010.png',
                  'images/samplelayouts/96wellplate_1509x1010.png']

parameterPerImage = [
    {
        "name": "4 Slide Carrier",
        "pixel_image_x": 1509,
        "pixel_image_y": 1010,
        "phys_dim_x": 127.7,
        "phys_dim_y": 85.5,
        "phys_offset_x": 0,
        "phys_offset_y": 0,
        "num_wells_x": 4,
        "num_wells_y": 1,
        "center_x": 63.85,
        "center_y": 42.75,
        "well_spacing_x": 31,
        "well_spacing_y": 0,
        "well_radius": 10,
        "filePath": 'images/samplelayouts/4slidecarrier_1509x1010.png'
    },
    {
        "name": "6 Well Plate",
        "pixel_image_x": 1509,
        "pixel_image_y": 1010,
        "phys_dim_x": 127.7,
        "phys_dim_y": 85.5,
        "phys_offset_x": 0,
        "phys_offset_y": 0,
        "num_wells_x": 3,
        "num_wells_y": 2,
        "center_x": 63.85,
        "center_y": 42.75,
        "well_spacing_x": 39,
        "well_spacing_y": 39,
        "well_radius": 17.5,
        "filePath": 'images/samplelayouts/6wellplate_1509x1010.png'
    },
    {
        "name": "12 Well Plate",
        "pixel_image_x": 1509,
        "pixel_image_y": 1010,
        "phys_dim_x": 127.7,
        "phys_dim_y": 85.5,
        "phys_offset_x": 0,
        "phys_offset_y": 0,
        "num_wells_x": 4,
        "num_wells_y": 3,
        "center_x": 63.85,
        "center_y": 42.75,
        "well_spacing_x": 26,
        "well_spacing_y": 26,
        "well_radius": 8.5,
        "filePath": 'images/samplelayouts/12wellplate_1509x1010.png'
    },
    {
        "name": "24 Well Plate",
        "pixel_image_x": 1509,
        "pixel_image_y": 1010,
        "phys_dim_x": 127.7,
        "phys_dim_y": 85.5,
        "phys_offset_x": 0,
        "phys_offset_y": 0,
        "num_wells_x": 6,
        "num_wells_y": 4,
        "center_x": 63.85,
        "center_y": 42.75,
        "well_spacing_x": 18.9,
        "well_spacing_y": 18.9,
        "well_radius": 6.5,
        "filePath": 'images/samplelayouts/24wellplate_1509x1010.png'
    },
    {
        "name": "96 Well Plate",
        "pixel_image_x": 1509,
        "pixel_image_y": 1010,
        "phys_dim_x": 127.7,
        "phys_dim_y": 85.5,
        "phys_offset_x": 0,
        "phys_offset_y": 0,
        "num_wells_x": 12,
        "num_wells_y": 8,
        "center_x": 63.85,
        "center_y": 42.75,
        "well_spacing_x": 9,
        "well_spacing_y": 9,
        "well_radius": 4.5,
        "filePath": 'images/samplelayouts/96wellplate_1509x1010.png'
    }
]

for iWellplate in parameterPerImage:

    wellplate_json = generate_wellplate_json(
        name = iWellplate['name'],
        pixel_image_x = iWellplate['pixel_image_x'],
        pixel_image_y = iWellplate['pixel_image_y'],
        phys_dim_x = iWellplate['phys_dim_x'],
        phys_dim_y = iWellplate['phys_dim_y'],
        phys_offset_x = iWellplate['phys_offset_x'],
        phys_offset_y = iWellplate['phys_offset_y'],
        image_path = iWellplate['filePath'],
        num_wells_x = iWellplate['num_wells_x'],
        num_wells_y = iWellplate['num_wells_y'],
        center_x = iWellplate['center_x'],
        center_y = iWellplate['center_y'],
        well_spacing_x = iWellplate['well_spacing_x'],
        well_spacing_y = iWellplate['well_spacing_y'],
        well_radius = iWellplate['well_radius']
    )
    basePath = 'imswitch/_data/images/'

    json_file_path = os.path.join(basePath, f"{iWellplate['name'].replace(' ', '_').lower()}.json")

    with open(json_file_path, "w") as file:
        json.dump(wellplate_json, file, indent=4)

    json_file_path
