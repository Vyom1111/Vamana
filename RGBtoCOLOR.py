from webcolors import CSS3_HEX_TO_NAMES, hex_to_rgb
from scipy.spatial import KDTree
import numpy as np

def closest_color_name(rgb_tuple):
    css3_db = {hex_to_rgb(hex): name for hex, name in CSS3_HEX_TO_NAMES.items()}
    color_tree = KDTree(list(css3_db.keys()))
    dist, index = color_tree.query(rgb_tuple)
    closest_name = list(css3_db.values())[index]
    return closest_name

def blend_color_name(rgb_tuple):
    css3_db = {hex_to_rgb(hex): name for hex, name in CSS3_HEX_TO_NAMES.items()}
    color_tree = KDTree(list(css3_db.keys()))
    distances, indices = color_tree.query(rgb_tuple, k=2)
    
    if distances[0] < 10:  # Threshold for exact match
        return list(css3_db.values())[indices[0]]
    
    name1, name2 = list(css3_db.values())[indices[0]], list(css3_db.values())[indices[1]]
    return f"{name1}-{name2} Mix"

# Example Usage
rgb_value = (98.69081033389396, 180.59486062477598, 122.97044307811078)  # Input RGB color
print("Closest Named Color:", closest_color_name(rgb_value))
print("Blended Name:", blend_color_name(rgb_value))
