"""
Wellplate layout definitions and generator functions.
Provides pre-defined wellplate layouts and utilities to generate custom layouts.
"""
from typing import Dict, List, Any, Optional
from pydantic import BaseModel, Field


class WellDefinition(BaseModel):
    """Definition of a single well in a wellplate layout"""
    id: str = Field(..., description="Unique identifier for the well (e.g., 'A1', 'B2')")
    name: str = Field(..., description="Human-readable name for the well")
    x: float = Field(..., description="X coordinate of well center in micrometers")
    y: float = Field(..., description="Y coordinate of well center in micrometers")
    shape: str = Field(..., description="Shape of the well: 'circle' or 'rectangle'")
    radius: Optional[float] = Field(None, description="Radius for circular wells in micrometers")
    width: Optional[float] = Field(None, description="Width for rectangular wells in micrometers")
    height: Optional[float] = Field(None, description="Height for rectangular wells in micrometers")
    row: int = Field(..., description="Row index (0-based)")
    col: int = Field(..., description="Column index (0-based)")


class WellplateLayout(BaseModel):
    """Complete wellplate layout definition"""
    name: str = Field(..., description="Name of the layout")
    description: str = Field("", description="Description of the layout")
    rows: int = Field(..., description="Number of rows")
    cols: int = Field(..., description="Number of columns")
    well_spacing_x: float = Field(..., description="Spacing between well centers in X (micrometers)")
    well_spacing_y: float = Field(..., description="Spacing between well centers in Y (micrometers)")
    offset_x: float = Field(0, description="Offset from origin in X (micrometers)")
    offset_y: float = Field(0, description="Offset from origin in Y (micrometers)")
    wells: List[WellDefinition] = Field(default_factory=list, description="List of wells")


def generate_wellplate_layout(
    name: str,
    rows: int,
    cols: int,
    well_spacing_x: float,
    well_spacing_y: float,
    well_shape: str = "circle",
    well_radius: Optional[float] = None,
    well_width: Optional[float] = None,
    well_height: Optional[float] = None,
    offset_x: float = 0,
    offset_y: float = 0,
    description: str = ""
) -> WellplateLayout:
    """
    Generate a wellplate layout programmatically.
    
    Args:
        name: Name of the layout
        rows: Number of rows (e.g., 8 for 96-well plate)
        cols: Number of columns (e.g., 12 for 96-well plate)
        well_spacing_x: Spacing between well centers in X (micrometers)
        well_spacing_y: Spacing between well centers in Y (micrometers)
        well_shape: 'circle' or 'rectangle'
        well_radius: Radius for circular wells (micrometers)
        well_width: Width for rectangular wells (micrometers)
        well_height: Height for rectangular wells (micrometers)
        offset_x: Offset from origin in X (micrometers)
        offset_y: Offset from origin in Y (micrometers)
        description: Description of the layout
        
    Returns:
        WellplateLayout object
    """
    wells = []
    
    # Standard row naming: A, B, C, ...
    row_names = [chr(65 + i) for i in range(rows)]  # A=65 in ASCII
    
    for row in range(rows):
        for col in range(cols):
            well_id = f"{row_names[row]}{col + 1}"
            well_x = offset_x + col * well_spacing_x
            well_y = offset_y + row * well_spacing_y
            
            well = WellDefinition(
                id=well_id,
                name=well_id,
                x=well_x,
                y=well_y,
                shape=well_shape,
                radius=well_radius if well_shape == "circle" else None,
                width=well_width if well_shape == "rectangle" else None,
                height=well_height if well_shape == "rectangle" else None,
                row=row,
                col=col
            )
            wells.append(well)
    
    return WellplateLayout(
        name=name,
        description=description,
        rows=rows,
        cols=cols,
        well_spacing_x=well_spacing_x,
        well_spacing_y=well_spacing_y,
        offset_x=offset_x,
        offset_y=offset_y,
        wells=wells
    )


# Pre-defined standard wellplate layouts
def get_predefined_layouts() -> Dict[str, WellplateLayout]:
    """Get dictionary of pre-defined wellplate layouts"""
    layouts = {}
    
    # 96-well plate (standard SBS format)
    # 9mm spacing, 6.4mm diameter wells
    layouts["96-well-standard"] = generate_wellplate_layout(
        name="96-well Standard",
        description="Standard 96-well plate (SBS format)",
        rows=8,
        cols=12,
        well_spacing_x=9000,  # 9mm in micrometers
        well_spacing_y=9000,
        well_shape="circle",
        well_radius=3200,  # 6.4mm diameter = 3.2mm radius
        offset_x=0,
        offset_y=0
    )
    
    # 384-well plate (standard SBS format)
    # 4.5mm spacing, 3.6mm diameter wells
    layouts["384-well-standard"] = generate_wellplate_layout(
        name="384-well Standard",
        description="Standard 384-well plate (SBS format)",
        rows=16,
        cols=24,
        well_spacing_x=4500,  # 4.5mm in micrometers
        well_spacing_y=4500,
        well_shape="circle",
        well_radius=1800,  # 3.6mm diameter = 1.8mm radius
        offset_x=0,
        offset_y=0
    )
    
    # 24-well plate
    # 19.3mm spacing, 15.6mm diameter wells
    layouts["24-well-standard"] = generate_wellplate_layout(
        name="24-well Standard",
        description="Standard 24-well plate",
        rows=4,
        cols=6,
        well_spacing_x=19300,
        well_spacing_y=19300,
        well_shape="circle",
        well_radius=7800,
        offset_x=0,
        offset_y=0
    )
    
    # 6-well plate
    # 39.1mm spacing, 34.8mm diameter wells
    layouts["6-well-standard"] = generate_wellplate_layout(
        name="6-well Standard",
        description="Standard 6-well plate",
        rows=2,
        cols=3,
        well_spacing_x=39100,
        well_spacing_y=39100,
        well_shape="circle",
        well_radius=17400,
        offset_x=0,
        offset_y=0
    )
    
    # Custom histology slide layout (4 samples)
    layouts["histology-4x"] = generate_wellplate_layout(
        name="Histology 4x Sample",
        description="Layout for 4 histology samples on a slide",
        rows=2,
        cols=2,
        well_spacing_x=30000,
        well_spacing_y=30000,
        well_shape="rectangle",
        well_width=25000,
        well_height=25000,
        offset_x=0,
        offset_y=0
    )
    
    return layouts


def get_layout_by_name(layout_name: str, **params) -> Optional[WellplateLayout]:
    """
    Get a wellplate layout by name, with optional parameter overrides.
    
    Args:
        layout_name: Name of the pre-defined layout or 'custom'
        **params: Override parameters (offset_x, offset_y, etc.)
        
    Returns:
        WellplateLayout object or None if not found
    """
    layouts = get_predefined_layouts()
    
    if layout_name == "custom":
        # Generate custom layout from parameters
        required_params = ['rows', 'cols', 'well_spacing_x', 'well_spacing_y', 'well_shape']
        if not all(p in params for p in required_params):
            return None
        return generate_wellplate_layout(**params)
    
    # Get pre-defined layout
    layout = layouts.get(layout_name)
    if not layout:
        return None
    
    # Apply parameter overrides if provided
    if params:
        layout_dict = layout.dict()
        layout_dict.update(params)
        
        # Regenerate wells if offset changed
        if 'offset_x' in params or 'offset_y' in params:
            layout = generate_wellplate_layout(
                name=layout_dict['name'],
                rows=layout_dict['rows'],
                cols=layout_dict['cols'],
                well_spacing_x=layout_dict['well_spacing_x'],
                well_spacing_y=layout_dict['well_spacing_y'],
                well_shape=layout_dict['wells'][0].shape if layout_dict['wells'] else 'circle',
                well_radius=layout_dict['wells'][0].radius if layout_dict['wells'] and layout_dict['wells'][0].radius else None,
                well_width=layout_dict['wells'][0].width if layout_dict['wells'] and layout_dict['wells'][0].width else None,
                well_height=layout_dict['wells'][0].height if layout_dict['wells'] and layout_dict['wells'][0].height else None,
                offset_x=layout_dict['offset_x'],
                offset_y=layout_dict['offset_y'],
                description=layout_dict['description']
            )
        else:
            layout = WellplateLayout(**layout_dict)
    
    return layout
