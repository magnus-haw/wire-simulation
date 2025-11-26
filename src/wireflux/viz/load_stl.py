import pyvista as pv

def add_stl_to_plotter(plotter, path, color=(0.8, 0.8, 0.8),
                       translate=None, rotate=None, scale=None,
                       smooth=True, name=None):
    """
    Add an STL mesh to an existing PyVista plotter.

    Arguments
    ---------
    plotter : pv.Plotter
        The plotter to which the mesh will be added.
    path : str
        The STL file path.
    color : tuple
        RGB color (0–1 range).
    translate : tuple or list or None
        Optional translation (dx, dy, dz).
    rotate : dict or None
        Optional rotation spec, e.g.:
            rotate={"angle": 30, "axis": "z"}
            rotate={"angle": 45, "axis": (1,0,0)}
    scale : tuple or float or None
        Optional uniform or per-axis scaling.
    smooth : bool
        If True, compute normals for smooth shading.
    name : str or None
        Optional name for the mesh in the plotter.

    Returns
    -------
    plotter : pv.Plotter
        The same plotter object (to allow chaining).
    mesh : pv.PolyData
        The loaded mesh (in case user wants to manipulate it further).
    """
    # Load STL
    mesh = pv.read(path)

    # Optional smoothing (computes normals)
    if smooth:
        mesh = mesh.compute_normals(auto_orient_normals=True)

    # Optional transforms
    if translate is not None:
        mesh.translate(translate, inplace=True)

    if rotate is not None:
        angle = rotate.get("angle", 0)
        axis = rotate.get("axis", "z")
        mesh.rotate(angle=angle, axis=axis, inplace=True)

    if scale is not None:
        mesh.scale(scale, inplace=True)

    # Add to plotter
    plotter.add_mesh(
        mesh,
        color=color,
        smooth_shading=False,
        show_edges=False,
        specular=0.1,
        ambient=0.2,
        name=name,
    )

    return plotter, mesh
