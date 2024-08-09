import pytest
import sys
sys.path.insert(0,"../")
sys.path.insert(0,"./")
from terrain_analysis import *
from pylint.lint import Run
from pylint.reporters import CollectingReporter
from dataclasses import asdict
import numpy as np
import pandas as pd

class TestTerrainAnalysis():
    
    def test_convert_rasterio(self):

        import rasterio

        template = rasterio.open("test/data/raster_template.tif")
        data = np.zeros(template.shape)

        data_as_rasterio = convert_to_rasterio(data, template)

        assert type(data_as_rasterio) == rasterio.io.DatasetReader
        assert np.array_equal(data_as_rasterio.read(1), data)
        
    
    def test_extract_from_raster(self):

        import geopandas as gpd

        template = rasterio.open("test/data/raster_template.tif")
        point = gpd.read_file("test/data/test_point.shp")
        geom_sample = list(point.geometry)
        values = extract_values_from_raster(template, geom_sample)
        assert len(values) == 2
        assert values[0] == pytest.approx(2509.6870)
        assert values[1] == pytest.approx(2534.5088)

    def test_make_classifier(self):

        import sklearn
        
        test_data =  np.random.normal(size=20)
        data = {
            "x1": test_data,
            "x2": test_data * 2.45,
            "y": [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]
        }
        df = pd.DataFrame(data)
        classifier = make_classifier(df.drop('y',axis=1),df['y'])
        assert type(classifier) == sklearn.ensemble._forest.RandomForestClassifier
        assert classifier.n_classes_ == 2

    def test_create_dataframe(self):

        import geopandas as gpd

        template = rasterio.open("test/data/raster_template.tif")
        point = gpd.read_file("test/data/test_point.shp")
        geom_sample = list(point.geometry)
        df = create_dataframe(template, template, template,
                              template, template, geom_sample,
                              0)
        assert type(df) == gpd.geodataframe.GeoDataFrame
        assert len(df) == 2
        assert np.array_equal(np.array(df.columns), np.array(['elev', 'fault', 'slope', 'LC', 'Geol', 'ls']))
        assert df['ls'].to_list() == [0,0]
        

    def test_lint(self):
        files =  ["terrain_analysis.py"]
        #pylint_options = ["--disable=line-too-long,import-error,fixme"]
        pylint_options = []

        report = CollectingReporter()
        result = Run(
                    files,
                    reporter=report,
                    exit=False,
                )
        score = result.linter.stats.global_note
        nErrors = len(report.messages)

        print("Score: " + str(score))
        line_format = "{path}:{line}:{column}: {msg_id}: {msg} ({symbol})"
        for error in report.messages:
            print(line_format.format(**asdict(error)))   

        assert score > 3
        assert score > 5
        assert score > 7
        assert score > 9
        assert nErrors < 500
        assert nErrors < 400
        assert nErrors < 250
        assert nErrors < 100
        assert nErrors < 50
        assert nErrors < 10
        assert nErrors == 0

class TestRegression():

    def test_regression(self):

        from subprocess import run
        import os
        import rasterio

        result = run(["python3","terrain_analysis.py",
                                "--topography",
                                "data/AW3D30.tif",
                                "--geology",
                                "data/Geology.tif",
                                "--landcover",
                                "data/Landcover.tif",
                                "--faults",
                                "data/Confirmed_faults.shp",
                                "data/landslides.shp",
                                "test.tif"], capture_output=True, check=True)
        assert len(result.stdout) < 25

        raster = rasterio.open("test.tif")
        values = raster.read(1)
        assert values.max() <= 1
        assert values.min() >= 0
        os.remove("test.tif")
        

    def test_regression_verbose(self):

        from subprocess import run
        import os
        import rasterio

        result = run(["python3","terrain_analysis.py",
                                "--topography",
                                "data/AW3D30.tif",
                                "--geology",
                                "data/Geology.tif",
                                "--landcover",
                                "data/Landcover.tif",
                                "--faults",
                                "data/Confirmed_faults.shp",
                                "data/landslides.shp",
                                "--v",
                                "test.tif"], capture_output=True, check=True)
        assert len(result.stdout) > 25

        raster = rasterio.open("test.tif")
        values = raster.read(1)
        assert values.max() <= 1
        assert values.min() >= 0
        os.remove("test.tif")



