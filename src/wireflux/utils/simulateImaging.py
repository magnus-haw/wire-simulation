import numpy as np
import os
from scipy.interpolate import CubicSpline
from scipy.interpolate import PchipInterpolator
from ..utils.smooth import smooth3DVectors, smooth
from .wires import Wire
import matplotlib.pyplot as plt


# ------------------------------
#  Camera setup
# ------------------------------
# Inputs:
#   pv       plotter        - pyVista instance, for visualizing 3D scene
#   np.array focus          - Vector pointing to point-of-focus, the point at the center of the camera image, in m
#   np.array pos            - Vector pointing to the position of the camera, in m
#   int      focal_length   - Focal length of camera in mm, defaults to 50mm.
#   int      Fnum           - F-number of camera, representing real aperture width by focal_length/Fnum
#   float    ratio          - Ratio of camera sensor horizontal to vertical
#   float    sensorDiag     - Diagonal of sensor
#   float    roll           - Roll about camera axis, in degrees
#   bool     useDoF         - Set frustum cutoffs at estimated depth-of-field (visualize only those parts of the model that would be in focus)
#   float    zoom           - Magnification of camera, as a factor
# Outputs:
#   None
def setup_camera(plotter, focus, pos, focal_length=50, Fnum=8, ratio=3/2, sensorDiag=43.3, roll=0, useDoF=True, zoom=1):
    sceneCamera = pv.Camera()   # Instance of camera

    sceneCamera.disable_parallel_projection()   # Ensure images have perspective
    sceneCamera.position    = pos     # Camera position in world coords
    sceneCamera.focal_point = focus   # Object position in world coords 
    
    imageVector = pos - focus
    imageDist   = np.mag(imageVector)   # Distance from camera position to focus of image NOTE: Could also use sceneCamera.distance for this value -JQM20260310
 
    circleOfConfusion = np.max(focal_length/1000, imageDist/1000)    # Arbitrary; a good acceptable circle of confusion is typically ~1000x smaller than focus (i.e. 0.05mm for 50mm focus)
    if useDoF: depthOfField = 2*np.square(np.linalg.norm(imageVector))*Fnum/np.square(focal_length/1000)   # Estimate of DoF given camera params
    else: depthOfField = imageDist
        
    sceneCamera.view_angle = 2*np.atan(2*focal_length/(ratio*sensorDiag/np.sqrt(1+ratio**2)))   # Horizontal view angle from triangle formed by sensor horizontal and focal lengths
    sceneCamera.clipping_range = (imageDist-depthOfField, imageDist+depthOfField)   # Camera frustum extends from one DoF before object to one DoF after object
    sceneCamera.roll = roll
    sceneCamera.zoom(zoom)
    
    return




# ------------------------------
#  Take Photo
# ------------------------------
# Inputs:
#   pv       plotter        - Pyvista instance
#   float    resolution     - Resolution in pixels (say 4.08Mpix, or similar)
#   float    ratio          - Sensor ratio; typically 3:2 horizontal to vertical for full size (35mm) format
#   float    sensorDiag     - Diagonal dimension of sensor, for getting absolute pixel values
#   bool     transparency   - Whether or not to make the background transparent; useful for overlay
#   string   filename       - Name of photo to save
# Outputs:
#   np.array photo   - Simulated output photo
def take_photo(plotter, resolution, ratio=3/2, sensorDiag=43.3, transparent_background=False, filename='frame', return_img=False):
    exportDir = 'Frames'
    operantDir = os.getcwd()
    print(operantDir) # DEBUG

    # Check if export dir exists, make one if it does not
    if (exportDir in os.listdir(operantDir)): continue
    else: os.mkdir(operantDir+'/'+exportDir)

    x_pixels = resolution*(ratio*sensorDiag/np.sqrt(1+ratio**2))   # Number of pixels in x, given resolution in megapixels, sensor diagonal, and ratio
    y_pixels = resolution*(sensorDiag/np.sqrt(1+ratio**2))         # Number of pixels in y

    # Save screenshot
    photo = plotter.screenshot(operantDir+'/'+exportDir+filename, 
                               transparent_background=transparent_background, return_img=return_img, 
                               window_size=(x_pixels,y_pixels) , scale=1)
    
    if return_img: return photo
    else: return