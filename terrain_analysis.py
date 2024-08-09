import argparse
import numpy as np
import rasterio
import pandas as pd
import geopandas as gpd
import sklearn
from sklearn.ensemble import RandomForestClassifier


def convert_to_rasterio(raster_data, template_raster):
    '''
    Reads the first layer (band) of the rasters data and copies
    using '[:]' modifying 'raster_data' array rather than
    generating a new one.
'''  
    raster_data[:] = template_raster.read(1)
   
    return template_raster

def extract_values_from_raster(raster, shape_object):
    
    # creates a list of coordinate pairs using the shape objects
    coord_pairs = [(shape.x, shape.y) for shape in shape_object]
    
    # samples the raster at provided coordinates
    values = raster.sample(coord_pairs)
    
    # converts 'values' into a list
    value_list = []
    for value_sample in values:
        value_list.append(value_sample[0])
    
    return value_list

# dem (Digital Elevation Model)
def calculate_slope(dem, x_value, y_value):
    
    # calculates gradient in direction of x and y
    x, y = np.gradient(dem, x_value, y_value)
    
    # calculates slope length and converts to angle in degrees 
    h_slope_degrees = np.arctan(np.sqrt(x**2 + y**2) * (180 / np.pi))
    
    return h_slope_degrees 
            
def distance_from_fault_raster(fault):
    distance = proximity(topo)
    
def make_classifier(x, y, verbose=False):
    
    classifier = RandomForestClassifier(verbose=verbose)
    classifier.fit(x,y)
    
    return classifier

def make_prob_raster_data(topo, geo, lc, dist_fault, slope, classifier):

    return

def create_dataframe(topo, geo, lc, dist_fault, slope, shape, landslides):
    
    return gpd.geodataframe.GeoDataFrame(pd.DataFrame(
        {'elev':extract_values_from_raster(topo, shape),
         'fault':extract_values_from_raster(dist_fault, shape),
         'slope':extract_values_from_raster(slope, shape),
         'LC':extract_values_from_raster(lc, shape),
         'Geol':extract_values_from_raster(geo, shape),
         'ls':landslides}
        ))

def main():


    parser = argparse.ArgumentParser(
                     prog="Landslide hazard using ML",
                     description="Calculate landslide hazards using simple ML",
                     epilog="Copyright 2024, Jon Hill"
                     )
    parser.add_argument('--topography',
                    required=True,
                    help="topographic raster file")
    parser.add_argument('--geology',
                    required=True,
                    help="geology raster file")
    parser.add_argument('--landcover',
                    required=True,
                    help="landcover raster file")
    parser.add_argument('--faults',
                    required=True,
                    help="fault location shapefile")
    parser.add_argument("--landslides",
                    help="the landslide location shapefile")
    parser.add_argument("--output",
                    help="the output raster file")
    parser.add_argument('-v', '--verbose',
                    action='store_true',
                    default=False,
                    help="Print progress")

    args = parser.parse_args()

    topography = rasterio.open(args.topography)
    topography_data = np.zeros(topography.shape)
    convert_to_rasterio(topography_data, topography)
    
if __name__ == '__main__':
    main()