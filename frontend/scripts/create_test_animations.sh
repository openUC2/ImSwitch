#!/bin/bash
# Script to create placeholder GIF animations for acceptance testing

# Create animations directory if it doesn't exist
mkdir -p ../public/assets/animations

cd ../public/assets/animations

# Function to create a simple placeholder GIF
create_placeholder() {
    local filename=$1
    local text=$2
    local color=$3
    
    # Create a simple static placeholder (requires ImageMagick)
    # If ImageMagick is not available, you'll need to create these manually
    if command -v convert &> /dev/null; then
        convert -size 300x200 "xc:${color}" \
                -pointsize 16 \
                -fill white \
                -gravity center \
                -draw "text 0,0 '${text}'" \
                "${filename}"
        echo "Created: ${filename}"
    else
        echo "ImageMagick not found. Please create ${filename} manually."
        echo "  Text: ${text}"
        echo "  Suggested size: 300x200px"
    fi
}

# Create motion animations
create_placeholder "home_x.gif" "Homing X Axis\n(Motor moving left)" "#4A90E2"
create_placeholder "home_y.gif" "Homing Y Axis\n(Motor moving back)" "#4A90E2"
create_placeholder "home_z.gif" "Homing Z Axis\n(Motor moving down)" "#4A90E2"
create_placeholder "move_absolute.gif" "Moving to Position\nX:2000 Y:2000 Z:2000" "#7B68EE"
create_placeholder "move_x_plus.gif" "Moving Right\n+1000 µm" "#50C878"
create_placeholder "move_x_minus.gif" "Moving Left\n-1000 µm" "#FF6B6B"

# Create lighting animations
create_placeholder "light_on.gif" "Light Source\nTurning ON" "#FFD700"
create_placeholder "light_off.gif" "Light Source\nTurning OFF" "#696969"

# Create autofocus animation
create_placeholder "autofocus.gif" "Autofocus Running\nFinding optimal focus" "#9370DB"

echo ""
echo "Placeholder animations created!"
echo ""
echo "Note: These are static placeholders. For better UX, replace them with:"
echo "  - Animated GIFs showing actual microscope movement"
echo "  - Arrows indicating direction of motion"
echo "  - Simple looping animations created in any GIF editor"
echo ""
echo "Recommended tools for creating animations:"
echo "  - GIMP (free, cross-platform)"
echo "  - ezgif.com (online editor)"
echo "  - Record actual microscope footage and convert to GIF"
