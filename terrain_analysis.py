import argparse
import numpy as np
import rasterio
import geopandas as gpd

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
        value_sample[0]
    
    return value_list
    
    
def make_classifier(x, y, verbose=False):

    return

def make_prob_raster_data(topo, geo, lc, dist_fault, slope, classifier):

    return

def create_dataframe(topo, geo, lc, dist_fault, slope, shape, landslides):

    return


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