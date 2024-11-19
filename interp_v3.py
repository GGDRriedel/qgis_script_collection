# -*- coding: utf-8 -*-

"""
***************************************************************************
*                                                                         *
*   This program is free software; you can redistribute it and/or modify  *
*   it under the terms of the GNU General Public License as published by  *
*   the Free Software Foundation; either version 2 of the License, or     *
*   (at your option) any later version.                                   *
*                                                                         *
***************************************************************************
"""

from qgis.PyQt.QtCore import QCoreApplication,QVariant
from qgis.core import (QgsProcessing,
                       QgsFeatureSink,
                       QgsProcessingException,
                       QgsProcessingAlgorithm,
                       QgsProcessingParameterFeatureSource,
                       QgsProcessingParameterRasterLayer,
                       QgsProcessingParameterString,
                       QgsProcessingParameterFile,
                       QgsProcessingParameterFileDestination,
                       QgsCoordinateReferenceSystem,
                       QgsPointXY,
                       QgsFields,
                       QgsCoordinateTransform,
                       QgsProcessingParameterFeatureSink,
                       QgsField,
                       QgsFeature,
                       QgsGeometry,
                       QgsMessageLog)
from qgis import processing

# calculation imports
from scipy.interpolate import interp1d
import pandas as pand
import numpy as np
from tabulate import tabulate

def to_prn(df, fname):
            #https://stackoverflow.com/questions/16490261/python-pandas-write-dataframe-to-fixed-width-file-to-fwf
            content = tabulate(df.values.tolist(), list(df.columns), tablefmt="plain",\
                       intfmt="10d",floatfmt=("10.2f"))
            with open(fname, "w") as file: file.write(content)


class ExampleProcessingAlgorithm(QgsProcessingAlgorithm):
    """
    This is an example algorithm that takes a vector layer and
    creates a new identical one.

    It is meant to be used as an example of how to create your own
    algorithms and explain methods and variables used to do it. An
    algorithm like this will be available in all elements, and there
    is not need for additional work.

    All Processing algorithms should extend the QgsProcessingAlgorithm
    class.
    """

    # Constants used to refer to parameters and outputs. They will be
    # used when calling the algorithm from another algorithm, or when
    # calling from the QGIS console.

    INPUT = 'INPUT'
    INPUT_DEM = 'INPUT_DEM'
    INPUT_STATION_NAME="INPUT_STATION_NAME"
    INPUT_HEIGHT_NAME="INPUT_HEIGHT_NAME"
    OUTPUT = 'OUTPUT'
    OUTPUT_INTERP= 'OUTPUT_INTERP'
    OUTPUT_PRN= 'Output_Prn'
    
    
    
    
    
    
    
    def tr(self, string):
        """
        Returns a translatable string with the self.tr() function.
        """
        return QCoreApplication.translate('Processing', string)

    def createInstance(self):
        return ExampleProcessingAlgorithm()

    def name(self):
        """
        Returns the algorithm name, used for identifying the algorithm. This
        string should be fixed for the algorithm, and must not be localised.
        The name should be unique within each provider. Names should contain
        lowercase alphanumeric characters only and no spaces or other
        formatting characters.
        """
        return 'dem_sampler_and_interp_V3'

    def displayName(self):
        """
        Returns the translated algorithm name, which should be used for any
        user-visible display of the algorithm name.
        """
        return self.tr('Dem Sampling and Interpolator Script V3')

    def group(self):
        """
        Returns the name of the group this algorithm belongs to. This string
        should be localised.
        """
        return self.tr('Coordinate Operations')

    def groupId(self):
        """
        Returns the unique ID of the group this algorithm belongs to. This
        string should be fixed for the algorithm, and must not be localised.
        The group id should be unique within each provider. Group id should
        contain lowercase alphanumeric characters only and no spaces or other
        formatting characters.
        """
        return 'coordinate_operations'

    def shortHelpString(self):
        """
        Returns a localised short helper string for the algorithm. This string
        should provide a basic description about what the algorithm does and the
        parameters and outputs associated with it..
        """
        return self.tr('Samples a DEM at the given points and does a 3D (!) distance interpolation ' \
        'Adds one layer with just sampled points in a new attribute that is set in the "Output Layer" and adds another layer with'\
        ' interpolated points according to the station numbering given in the station field. \n'\
        'Station FiePld needs to be the name of the attribute column that contains the station numbering \n\n'\
        'Also Writes a formated .prn file for the chosen prn file destination'\
        '\n\n The station distance is set automatically through the station numbering. \n'\
        'e.g. two stations named 1 and 10 at a distance of 5 meters to each other will generate 10 interpolated points every 0.5 Meters'
       )

    def initAlgorithm(self, config=None):
        """
        Here we define the inputs and output of the algorithm, along
        with some other properties.
        """

        # We add the input vector features source. It can have any kind of
        # geometry.
        self.addParameter(
            QgsProcessingParameterFeatureSource(
                self.INPUT,
                self.tr('Input layer'),
                [QgsProcessing.TypeVectorAnyGeometry]
            )
        )
        self.addParameter(
            QgsProcessingParameterRasterLayer(
                self.INPUT_DEM, 
                self.tr('DEM Raster layer'),
                None,False
            )
        )    
        
        self.addParameter(
            QgsProcessingParameterString(
                self.INPUT_STATION_NAME,
                self.tr('Station Field Name')
            )
        )
        self.addParameter(
            QgsProcessingParameterString(
                self.INPUT_HEIGHT_NAME,
                self.tr('Height Field Name'),
                defaultValue="Ortho Height"
            )
        )
        
        
                #prn folder
        self.addParameter(
            QgsProcessingParameterFileDestination(
                self.OUTPUT_PRN,
                self.tr('Output Interpolation Prn File')
                )
        )
    
        # We add a feature sink in which to store our processed features (this
        # usually takes the form of a newly created vector layer when the
        # algorithm is run in QGIS).
        self.addParameter(
            QgsProcessingParameterFeatureSink(
                self.OUTPUT,
                self.tr('Output layer')
            )
        )
    
        #output for interpolation
        self.addParameter(
            QgsProcessingParameterFeatureSink(
                self.OUTPUT_INTERP,
                self.tr('Interpolated Output layer')
            )
        )
        
        
        
    def processAlgorithm(self, parameters, context, feedback):
        
        
        """
        Here is where the processing itself takes place.
        """

        # Retrieve the feature source and sink. The 'dest_id' variable is used
        # to uniquely identify the feature sink, and must be included in the
        # dictionary returned by the processAlgorithm function.
        source = self.parameterAsSource(
            parameters,
            self.INPUT,
            context
        )
        
        rlayer=self.parameterAsRasterLayer(
            parameters,
            self.INPUT_DEM,
            context)
        
        station=self.parameterAsString(
            parameters,
            self.INPUT_STATION_NAME,
            context)
        
        height_name=self.parameterAsString(
            parameters,
            self.INPUT_HEIGHT_NAME,
            context)
        
        prn_file=self.parameterAsString(
            parameters,
            self.OUTPUT_PRN,
            context)
            
        # Send some information to the user
        feedback.pushInfo('DEM Extent is {}'.format(rlayer.extent().asWktCoordinates()))
        feedback.pushInfo('Chosen station field in data is {}'.format(station))
        feedback.pushInfo('Chosen height field in data is {}'.format(height_name))
        feedback.pushInfo('Prn file {}'.format(prn_file))
        
        
        # If source was not found, throw an exception to indicate that the algorithm
        # encountered a fatal error. The exception text can be any string, but in this
        # case we use the pre-built invalidSourceError method to return a standard
        # helper text for when a source cannot be evaluated
        if source is None:
            raise QgsProcessingException(self.invalidSourceError(parameters, self.INPUT))
        
        ### Processing!
        
         # Get the extent of the point layer
        source_layer_extent = source.sourceExtent()
        # Get the extent of the dem layer
        dem_layer_extent = rlayer.extent()
        
        # Check if the input layer is within the extent of the raster layer
        if not dem_layer_extent.contains(source_layer_extent):
            raise QgsProcessingException('Input layer is not within the extent of the raster layer.')
        else:
             feedback.pushInfo('Source Extent is {}'.format(source_layer_extent.asWktCoordinates()))
        features = source.getFeatures()
        
        # Get the raster data provider
        data_provider = rlayer.dataProvider()
         # Get the band index to sample
        band_index = 1
        
        crs = rlayer.crs()
        extent = rlayer.extent()

        # Get the coordinates of the points
        point_coords = []
        for feature in source.getFeatures():
            point = feature.geometry().asPoint()
            point_coords.append(point)

        # Sample the raster at the points
        sampled_values = []
        X=[]
        Y=[]
        for point in point_coords:
            # Transform the point to the raster's CRS
            point_crs = QgsCoordinateReferenceSystem(source.sourceCrs())
            point_crs_transform = QgsCoordinateTransform(point_crs, crs, context.project())
            point_transformed = point_crs_transform.transform(point)
            
            
            # Get the pixel value at the point
            x, y = (point.x(), point.y())
            X.append(x)
            Y.append(y)
            #value = data_provider.identify(QgsPointXY(x, y), band_index)
            value = data_provider.sample(QgsPointXY(x, y),band_index)
            sampled_values.append(value[0])
            #feedback.pushInfo('Appended {} {} {}'.format(x,y,value.results()[1]))
            
        #create new fields for our new layer
        #fields = QgsFields()
        #fields.append(QgsField('x', type=QVariant.Double))
        #fields.append(QgsField('y', type=QVariant.Double))
        #fields.append(QgsField('z', type=QVariant.Double))
        
        #source.dataProvider().addAttributes([QgsField('z',Qvariant.Double)])
        #source.updateFields()
        
        #copy the old structure and add the new one
        new_fields = source.fields()
        new_fields.append(QgsField('z', QVariant.Double))

        interp_fields=QgsFields()
        interp_fields.append(QgsField('GP', type=QVariant.Double))
        interp_fields.append(QgsField('X_int', type=QVariant.Double))
        interp_fields.append(QgsField('Y_int', type=QVariant.Double))
        interp_fields.append(QgsField('Z_int', type=QVariant.Double))
        interp_fields.append(QgsField('Z_Lidar', type=QVariant.Double))
        
        interp_fields.append(QgsField('Dist_2D', type=QVariant.Double))
        interp_fields.append(QgsField('Dist_3D', type=QVariant.Double))
        interp_fields.append(QgsField('Profm_2D', type=QVariant.Double))
        interp_fields.append(QgsField('Profm_3D', type=QVariant.Double))
        
        
        
        #sink for sampled points
        (sink, dest_id) = self.parameterAsSink(
            parameters,
            self.OUTPUT,
            context,
            new_fields,
            source.wkbType(),
            source.sourceCrs()
        )
        #sink for interpolated and sampled points
        (interp_sink, dest_id) = self.parameterAsSink(
            parameters,
            self.OUTPUT_INTERP,
            context,
            interp_fields,
            source.wkbType(),
            source.sourceCrs()
        )
        # If sink was not created, throw an exception to indicate that the algorithm
        # encountered a fatal error. The exception text can be any string, but in this
        # case we use the pre-built invalidSinkError method to return a standard
        # helper text for when a sink cannot be evaluated
        if sink is None:
            raise QgsProcessingException(self.invalidSinkError(parameters, self.OUTPUT))
        
        if interp_sink is None:
            raise QgsProcessingException(self.invalidSinkError(parameters, self.OUTPUT_INTERP))
        
        # Compute the number of steps to display within the progress bar and
        # get features from source
        total = 100.0 / source.featureCount() if source.featureCount() else 0
        features = source.getFeatures()

        for current, feature in enumerate(features):
            # Stop the algorithm if cancel button has been clicked
            if feedback.isCanceled():
                break
            #we do not add the original features, we are adding a copy of the old features
            new_feat = QgsFeature()
            new_feat.setFields(new_fields)
            new_feat.setGeometry(feature.geometry())
            #new_feat.setAttributes(feature.attributes())
            
            for current2,att in enumerate(feature.attributes()):
                #feedback.pushInfo( new_fields.field(current).name()+str(att))
                new_feat.setAttribute(new_fields.field(current2).name(),att)
            #feature.setFields(new_fields)
            
            #add the sampled dem here
            #feature['z'] = sampled_values[current]
            # Add a feature in the sink
            new_feat.setAttribute('z',sampled_values[current])
            
            #feedback.pushInfo(new_feat.attributes)
            sink.addFeature(new_feat, QgsFeatureSink.FastInsert)
            
            #sink.addFeature(feature, QgsFeatureSink.FastInsert)

            # Update the progress bar
            feedback.setProgress(int(current * total))

        # To run another Processing algorithm as part of this algorithm, you can use
        # processing.run(...). Make sure you pass the current context and feedback
        # to processing.run to ensure that all temporary layer outputs are available
        # to the executed algorithm, and that the executed algorithm can send feedback
        # reports to the user (and correctly handle cancellation and progress reports!)
        if False:
            buffered_layer = processing.run("native:buffer", {
                'INPUT': dest_id,
                'DISTANCE': 1.5,
                'SEGMENTS': 5,
                'END_CAP_STYLE': 0,
                'JOIN_STYLE': 0,
                'MITER_LIMIT': 2,
                'DISSOLVE': False,
                'OUTPUT': 'memory:'
            }, context=context, feedback=feedback)['OUTPUT']

        # Return the results of the algorithm. In this case our only result is
        # the feature sink which contains the processed features, but some
        # algorithms may return multiple feature sinks, calculated numeric
        # statistics, etc. These should all be included in the returned
        # dictionary, with keys matching the feature corresponding parameter
        # or output names.
        
        
        #### this is experimental
        # turn data into pandas dataframe
        #https://gis.stackexchange.com/questions/403081/attribute-table-into-pandas-dataframe-pyqgis
        #List all columns you want to include in the dataframe. I include all with:
        cols = [f.name() for f in source.fields()]
        #A generator to yield one row at a time
        datagen = ([f[col] for col in cols] for f in source.getFeatures())
            
        data = pand.DataFrame.from_records(data=datagen, columns=cols)
        
        #feedback.pushInfo(data.columns[1])
        data["X"]=X
        data["Y"]=Y
        data[station]=data[station].astype(str)
        data["station"]=data[station].str.extract(r'(\d+.\d+|\d+)').astype('float') #('(\d+)')
        data["station"]=data["station"].astype(float)
        data.sort_values(by="station",inplace=True)
        data["average_X_diff"]=data.X.diff(periods=-1)/data.station.diff(periods=-1)
        data["average_Y_diff"]=data.Y.diff(periods=-1)/data.station.diff(periods=-1)
        data["average_Z_diff"]=data[height_name].diff(periods=-1)/data.station.diff(periods=-1)
        data["dist"]=np.sqrt(data.X.diff(periods=-1)**2+data.Y.diff(periods=-1)**2+data[height_name].diff(periods=-1)**2)/np.abs(data.station.diff(periods=-1))
        data["ok"]=data.dist.pct_change().abs()>0.1
        if (data.dist.pct_change().abs()>0.1).any():
            errordata=data[data.ok==True]
            QgsMessageLog.logMessage(errordata.to_string())
            feedback.pushInfo('The following data points seem to have a distance that differs from the average, please check:')
            feedback.pushInfo(errordata[["station","X","Y","average_X_diff","average_Y_diff", "average_Z_diff","dist"]].to_string())
            feedback.pushInfo("- - - - - - - - - - - - -")
            #raise QgsProcessingException("Error")
            feedback.pushInfo("We will continue anyway, please fix !")
        
        #interpolation
        feedback.pushInfo("\n ###### Interpolating\n")
        start=data.station.iloc[0].astype(int)
        stop=data.station.iloc[-1].astype(int)
        targetgeophones=np.linspace(start,stop,stop-start+1)
        Xinterp=interp1d(data.station,data.X,fill_value="extrapolate")
        Yinterp=interp1d(data.station,data.Y,fill_value="extrapolate")
        Zinterp=interp1d(data.station,data[height_name],fill_value="extrapolate")
        
        new=pand.DataFrame()
        new["GP"]=targetgeophones
        new["X_int"]=Xinterp(targetgeophones)
        new["Y_int"]=Yinterp(targetgeophones)
        new["Z_int"]=Zinterp(targetgeophones)
        new["Dist_2D"] = ""
        new["Dist_3D"] = ""
        
        
        coord_list=new[["X_int","Y_int"]].values
        
        
        #sample dem at new values
        # Sample the raster at the points
        interp_sampled_values = []
        for point in coord_list:
            #transformation not necessary           
            
            # Get the pixel value at the point
            interp_x=point[0]
            interp_y=point[1]
            
            value = data_provider.sample(QgsPointXY(interp_x, interp_y), band_index)
            interp_sampled_values.append(value[0])
           # feedback.pushInfo('Appended {} {} {}'.format(interp_x,interp_y,value.results()[1]))
        new["Z_DGM"]=interp_sampled_values
        
        #calculate the distances
        for i in range(1,len(new["GP"])):
        #  print(i)
            vector=new.iloc[i-1:i+1,1:4].values
            new["Dist_2D"][i]=np.sqrt((new["X_int"][i]-new["X_int"][i-1])**2 + \
                                (new["Y_int"][i]-new["Y_int"][i-1])**2)
            
            new["Dist_3D"][i]=np.sqrt((new["X_int"][i]-new["X_int"][i-1])**2 + \
                               (new["Y_int"][i]-new["Y_int"][i-1])**2 +\
                               (new["Z_DGM"][i]-new["Z_DGM"][i-1])**2   )
        #set first ones to 0
        new["Dist_2D"][0]=0
        new["Dist_3D"][0]=0    
        
        # calculate the profilemeters
        new["Profm_2D"]= new["Dist_2D"].cumsum()
        new["Profm_3D"]= new["Dist_3D"].cumsum()
        
        #put into interp_sink
        

        for counter, interp_z in enumerate(interp_sampled_values):
            # Stop the algorithm if cancel button has been clicked
            #feedback.pushInfo(str(counter))
            #feedback.pushInfo(str(new["X_int"][counter]))
            if feedback.isCanceled():
                break
            #we do not add the original features, we are adding a copy of the old features
            new_feat = QgsFeature()
            new_feat.setFields(interp_fields)
            #create geometry from point 
            interp_geometry=QgsGeometry.fromPointXY(QgsPointXY(new["X_int"][counter],new["Y_int"][counter]))
            new_feat.setGeometry(interp_geometry)
            # Add a feature in the sink
            
            xi=new["X_int"][counter].astype(float)
            #feedback.pushInfo(str(xi))
            new_feat.setAttribute('GP',float(new.iloc[counter,0]))
            new_feat.setAttribute('X_int',float(new.iloc[counter,1]))
            new_feat.setAttribute('Y_int',float(new["Y_int"][counter]))
            new_feat.setAttribute('Z_int',float(new["Z_int"][counter]))
            new_feat.setAttribute('Z_Lidar',interp_sampled_values[counter])
            
            new_feat.setAttribute('Dist_2D',float(new["Dist_2D"][counter]))
            new_feat.setAttribute('Dist_3D',float(new["Dist_3D"][counter]))
            new_feat.setAttribute('Profm_2D',float(new["Profm_2D"][counter]))
            new_feat.setAttribute('Profm_3D',float(new["Profm_3D"][counter]))
           
           
            
   
            
            
            #feedback.pushInfo(new_feat.attributes)
            interp_sink.addFeature(new_feat, QgsFeatureSink.FastInsert)
            
            #sink.addFeature(feature, QgsFeatureSink.FastInsert)

            # Update the progress bar
            feedback.setProgress(int(current * total))
        to_prn(new,prn_file)
        
        return {self.OUTPUT: dest_id}
