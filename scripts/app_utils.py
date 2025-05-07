#!python3

# A collection of utilities that can be used with the Open Data Cube API.
#
# License: Apache 2.0

# Created for EASI Hub Case Studies notebooks, https://dev.azure.com/csiro-easi/easi-hub-public/_git/hub-notebooks


import numpy as np
from skimage import exposure
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.patheffects as PathEffects
import calendar


def animated_timeseries(ds, output_path, width_pixels=600, interval=200, 
                        bands=['red', 'green', 'blue'], percentile_stretch = (0.02, 0.98),
                        title=False, show_date=True, annotation_kwargs={},
                        time_dim = 'time', x_dim = 'x', y_dim = 'y'):
    
    """
    Takes an xarray time series and animates the data as a three-band (e.g. true or false colour) 
    animation, allowing changes in the landscape to be compared across time.
    
    Animations can be exported as .mp4 (ideal for Twitter/social media), .wmv (ideal for Powerpoint) and .gif 
    (ideal for all purposes, but can have large file sizes) format files, and customised to include titles and 
    date annotations or use specific combinations of input bands.  
    
    Modified from the work of: Robbi Bishop-Taylor, Sean Chua, Bex Dunn    
    
    :param ds: 
        An xarray dataset with multiple time steps (i.e. multiple observations along the `time` dimension).
        
    :param output_path: 
        A string giving the output location and filename of the resulting animation. File extensions of '.mp4', 
        '.wmv' and '.gif' are accepted.
    
    :param width_pixels:
        An integer defining the output width in pixels for the resulting animation. The height of the animation is
        set automatically based on the dimensions/ratio of the input xarray dataset. Defaults to 600 pixels wide.
        
    :param interval:
        An integer defining the milliseconds between each animation frame used to control the speed of the output
        animation. Higher values result in a slower animation. Defaults to 200 milliseconds between each frame. 
        
    :param bands:
        An optional list of either one or three bands to be plotted, all of which must exist in `ds`.
        Defaults to `['red', 'green', 'blue']`. 
        
    :param percentile_stretch:
        An optional tuple of two floats that can be used to clip one or three-band arrays by percentiles to produce 
        a more vibrant, visually attractive image that is not affected by outliers/extreme values. The default is 
        `(0.02, 0.98)` which is equivalent to xarray's `robust=True`.

    :param title: 
        An optional string or list of strings with a length equal to the number of timesteps in ds. This can be
        used to display a static title (using a string), or a dynamic title (using a list) that displays different
        text for each timestep. Defaults to False, which plots no title.
        
    :param show_date:
        An optional boolean that defines whether or not to plot date annotations for each animation frame. Defaults 
        to True, which plots date annotations based on ds.
        
    :param annotation_kwargs:
        An optional dict of kwargs for controlling the appearance of text annotations to pass to the matplotlib 
        `plt.annotate` function (see https://matplotlib.org/api/_as_gen/matplotlib.pyplot.annotate.html for options). 
        For example, `annotation_kwargs={'fontsize':20, 'color':'red', 'family':'serif'}. By default, text annotations 
        are plotted as white, size 25 mono-spaced font with a 4pt black outline in the top-right of the animation.   
        
    :param time_dim:
        An optional string allowing you to override the xarray dimension used for time. Defaults to 'time'.
    
    :param x_dim:
        An optional string allowing you to override the xarray dimension used for x coordinates. Defaults to 'x'.
    
    :param y_dim:
        An optional string allowing you to override the xarray dimension used for y coordinates. Defaults to 'y'.       
    """
       
    # Define function to convert xarray dataset to list three band numpy arrays
    def _ds_to_arrraylist(ds, bands, time_dim, x_dim, y_dim, percentile_stretch): 
        
        """
        Converts an xarray dataset to a list of numpy arrays for plt.imshow plotting
        """
        
        # Compute percents
        p_low, p_high = ds[bands].to_array().quantile(percentile_stretch).values

        array_list = []
        for i, timestep in enumerate(ds[time_dim]):

            # Select single timestep from the data array
            ds_i = ds[{time_dim: i}]

            # Get shape of array
            x = len(ds[x_dim])
            y = len(ds[y_dim])

            # Create new three band array                
            rawimg = np.zeros((y, x, 3), dtype=np.float32)

            # Add xarray bands into three dimensional numpy array
            for band, colour in enumerate(bands):

                rawimg[:, :, band] = ds_i[colour].values

            # Stretch contrast using percentile values
            img_toshow = exposure.rescale_intensity(rawimg, in_range=(p_low, p_high))

            array_list.append(img_toshow)
            
        return array_list, p_low, p_high
    
   
    ###############
    # Setup steps #
    ############### 

    # Test if all dimensions exist in dataset
    if time_dim in ds and x_dim in ds and y_dim in ds:        
        
        # First test if there are three bands, and that all exist in both datasets:
        if (len(bands) == 3) & all([(b in ds.data_vars) for b in bands]): 

            # Import xarrays as lists of three band numpy arrays
            imagelist, vmin, vmax = _ds_to_arrraylist(ds, bands=bands, 
                                                      time_dim=time_dim, x_dim=x_dim, y_dim=y_dim, 
                                                      percentile_stretch=percentile_stretch)
        
            # Get time, x and y dimensions of dataset and calculate width vs height of plot
            timesteps = len(ds[time_dim])    
            width = len(ds[x_dim])
            height = len(ds[y_dim])
            width_ratio = float(width) / float(height)
            height = 10.0 / width_ratio

            # If title is supplied as a string, multiply out to a list with one string per timestep.
            # Otherwise, use supplied list for plot titles.
            if isinstance(title, str) or isinstance(title, bool):
                title_list = [title] * timesteps 
            else:
                title_list = title

            # Set up annotation parameters that control font etc. The nested dict structure sets default 
            # values which can be overwritten/customised by the manually specified `annotation_kwargs`
            annotation_kwargs = dict({'xy': (1, 1), 'xycoords':'axes fraction', 
                                      'xytext':(-5, -5), 'textcoords':'offset points', 
                                      'horizontalalignment':'right', 'verticalalignment':'top', 
                                      'fontsize':25, 'color':'white', 
                                      'path_effects':[PathEffects.withStroke(linewidth=4, foreground='black')]},
                                      **annotation_kwargs)

            ###################
            # Initialise plot #
            ################### 
            
            # Set up figure
            fig, ax1 = plt.subplots(ncols=1) 
            fig.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=0, hspace=0)
            fig.set_size_inches(10.0, height, forward=True)
            ax1.axis('off')

            # Initialise axesimage objects to be updated during animation, setting extent from dims
            extents = [float(ds[x_dim].min()), float(ds[x_dim].max()), 
                       float(ds[y_dim].min()), float(ds[y_dim].max())]
            im = ax1.imshow(imagelist[0], extent=extents)

            # Initialise annotation objects to be updated during animation
            t = ax1.annotate('', **annotation_kwargs) 
    

            ########################################
            # Create function to update each frame #
            ########################################

            # Function to update figure
            def update_figure(frame_i):            
            
                # If possible, extract dates from time dimension
                try:

                    # Get human-readable date info (e.g. "16 May 1990")
                    ts = ds[time_dim][{time_dim:frame_i}].dt
                    year = ts.year.item()
                    month = ts.month.item()
                    day = ts.day.item()
                    date_string = '{} {} {}'.format(day, calendar.month_abbr[month], year)
                    
                except:
                    
                    date_string = ds[time_dim][{time_dim:frame_i}].values.item()

                # Create annotation string based on title and date specifications:
                title = title_list[frame_i]
                if title and show_date:
                    title_date = '{}\n{}'.format(date_string, title)
                elif title and not show_date:
                    title_date = '{}'.format(title)
                elif show_date and not title:
                    title_date = '{}'.format(date_string)           
                else:
                    title_date = ''

                # Update figure for frame
                im.set_array(imagelist[frame_i])
                t.set_text(title_date) 

                # Return the artists set
                return [im, t]


            ##############################
            # Generate and run animation #
            ##############################

            # Generate animation
            print('Generating {} frame animation'.format(timesteps))
            ani = animation.FuncAnimation(fig, update_figure, frames=timesteps, interval=interval, blit=True)

            # Export as either MP4 or GIF
            if output_path[-3:] == 'mp4':
                print('    Exporting animation to {}'.format(output_path))
                ani.save(output_path, dpi=width_pixels / 10.0)

            elif output_path[-3:] == 'wmv':
                print('    Exporting animation to {}'.format(output_path))
                ani.save(output_path, dpi=width_pixels / 10.0, 
                         writer=animation.FFMpegFileWriter(fps=1000 / interval, bitrate=4000, codec='wmv2'))

            elif output_path[-3:] == 'gif':
                print('    Exporting animation to {}'.format(output_path))
                ani.save(output_path, dpi=width_pixels / 10.0, writer='imagemagick')

            else:
                print('    Output file type must be either .mp4, .wmv or .gif')
                
            return ani

        else:        
            print('Please select three bands that all exist in the input dataset')  

    else:
        print('At least one x, y or time dimension does not exist in the input dataset. Please use the `time_dim`,' \
              '`x_dim` or `y_dim` parameters to override the default dimension names used for plotting') 



from pyproj import Transformer
import math
import folium

# Borrowed from https://github.com/GeoscienceAustralia/dea-notebooks/blob/develop/Tools/dea_tools/plotting.py
def display_map(x, y, crs='EPSG:4326', margin=-0.5, zoom_bias=0):
    """ 
    Given a set of x and y coordinates, this function generates an 
    interactive map with a bounded rectangle overlayed on Google Maps 
    imagery.        
    
    Last modified: September 2019
    
    Modified from function written by Otto Wagner available here: 
    https://github.com/ceos-seo/data_cube_utilities/tree/master/data_cube_utilities
    
    Parameters
    ----------  
    x : (float, float)
        A tuple of x coordinates in (min, max) format. 
    y : (float, float)
        A tuple of y coordinates in (min, max) format.
    crs : string, optional
        A string giving the EPSG CRS code of the supplied coordinates. 
        The default is 'EPSG:4326'.
    margin : float
        A numeric value giving the number of degrees lat-long to pad 
        the edges of the rectangular overlay polygon. A larger value 
        results more space between the edge of the plot and the sides 
        of the polygon. Defaults to -0.5.
    zoom_bias : float or int
        A numeric value allowing you to increase or decrease the zoom 
        level by one step. Defaults to 0; set to greater than 0 to zoom 
        in, and less than 0 to zoom out.
        
    Returns
    -------
    folium.Map : A map centered on the supplied coordinate bounds. A 
    rectangle is drawn on this map detailing the perimeter of the x, y 
    bounds.  A zoom level is calculated such that the resulting 
    viewport is the closest it can possibly get to the centered 
    bounding rectangle without clipping it. 
    """

    # Convert each corner coordinates to lat-lon
    points = [ (x[0],y[0]), (x[1],y[0],), (x[0],y[1]), (x[1],y[1]) ]
    transformer = Transformer.from_crs(crs, 'EPSG:4326')
    tmp = np.array( list(transformer.itransform(points)) )
    all_longitude = tmp[:,0]; all_latitude = tmp[:,1]

    # Calculate zoom level based on coordinates
    lat_zoom_level = _degree_to_zoom_level(min(all_latitude),
                                           max(all_latitude),
                                           margin=margin) + zoom_bias
    lon_zoom_level = _degree_to_zoom_level(min(all_longitude),
                                           max(all_longitude),
                                           margin=margin) + zoom_bias
    zoom_level = min(lat_zoom_level, lon_zoom_level)

    # Identify centre point for plotting
    center = [np.mean(all_latitude), np.mean(all_longitude)]

    # Create map
    interactive_map = folium.Map(
        location=center,
        zoom_start=zoom_level,
        tiles="http://mt1.google.com/vt/lyrs=y&z={z}&x={x}&y={y}",
        attr="Google")

    # Create bounding box coordinates to overlay on map
    line_segments = [(all_latitude[0], all_longitude[0]),
                     (all_latitude[1], all_longitude[1]),
                     (all_latitude[3], all_longitude[3]),
                     (all_latitude[2], all_longitude[2]),
                     (all_latitude[0], all_longitude[0])]

    # Add bounding box as an overlay
    interactive_map.add_child(
        folium.features.PolyLine(locations=line_segments,
                                 color='red',
                                 opacity=0.8))

    # Add clickable lat-lon popup box
    interactive_map.add_child(folium.features.LatLngPopup())

    return interactive_map

def _degree_to_zoom_level(l1, l2, margin=0.0):
    
    """
    Helper function to set zoom level for `display_map`
    """
    
    degree = abs(l1 - l2) * (1 + margin)
    zoom_level_int = 0
    if degree != 0:
        zoom_level_float = math.log(360 / degree) / math.log(2)
        zoom_level_int = int(zoom_level_float)
    else:
        zoom_level_int = 18
    return zoom_level_int

