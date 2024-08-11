"""Duncan Potter Landslide risk coursework"""

import argparse
import numpy as np
import rasterio
import pandas as pd
import geopandas as gpd
from sklearn.ensemble import RandomForestClassifier
from proximity import proximity


def convert_to_rasterio(raster_data, template_raster):
    '''
    Reads the first layer (band) of the rasters data and copies
    using '[:]' modifying 'raster_data' array rather than
    generating a new one.
    '''
    raster_data[:] = template_raster.read(1)

    return template_raster

def extract_values_from_raster(raster, shape_object):
    '''extracts values from the raster file'''

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
    '''Calulates slope using pythagoras' theorem'''
    # calculates gradient in direction of x and y
    x_data, y_data = np.gradient(dem, x_value, y_value)

    # calculates slope length and converts to angle in degrees
    h_slope_degrees = np.arctan(np.sqrt(x_data**2 + y_data**2) * (180 / np.pi))

    return h_slope_degrees

def distance_from_fault_raster(topo, fault):
    '''Generates the distance from slope raster using the proximity function'''
    dist_fault = proximity(topo, fault, 1)

    return dist_fault

def make_classifier(x_values, y_values, verbose=False):
    '''Generates a random forest classifier'''
    classifier = RandomForestClassifier(verbose=verbose)
    classifier.fit(x_values,y_values)

    return classifier

# def make_prob_raster_data(topo, geo, lc, dist_fault, slope, classifier):
   # return

def create_dataframe(topo, geo, landcover, dist_fault, slope, shape, landslides):


    return gpd.geodataframe.GeoDataFrame(pd.DataFrame(
        {'elev':extract_values_from_raster(topo, shape),
         'fault':extract_values_from_raster(dist_fault, shape),
         'slope':extract_values_from_raster(slope, shape),
         'LC':extract_values_from_raster(landcover, shape),
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

    topo = rasterio.open(args.topography)
    geo = rasterio.open(args.geology)
    landcover = rasterio.open(args.landcover)
    fault = gpd.read_file(args.faults)
    landslides = gpd.read_file(args.landslides)

    dem = topo.read(1)
    x_value, y_value = topo.res
    slope = calculate_slope(dem, x_value, y_value)


    topography_data = np.zeros(topo.shape)
    convert_to_rasterio(topography_data, topo)

    geology_data = np.zeros(geo.shape)
    convert_to_rasterio(geology_data, geo)

    landcover_data = np.zeros(landcover.shape)
    convert_to_rasterio(landcover_data, landcover)

    dist_fault_data = distance_from_fault_raster(topo, fault)

    dataframe = create_dataframe(topography_data, geology_data, landcover_data,
                          dist_fault_data, slope, fault, landslides)

    print(dataframe)

if __name__ == '__main__':
    main()
