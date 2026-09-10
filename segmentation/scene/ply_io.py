"""storePly(), extracted standalone from COB-GS's scene/dataset_readers.py.

That module's top-level imports pull in scene.gaussian_model (and, via it,
the simple-knn compiled CUDA extension) purely for unrelated 3DGS
scene-loading code -- storePly() itself has no such dependency, so it's
kept here on its own to avoid dragging that chain into the segmentation
stage.
"""
import numpy as np
from plyfile import PlyData, PlyElement


def storePly(path, xyz, rgb):
    # Define the dtype for the structured array
    dtype = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
             ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4'),
             ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]

    normals = np.zeros_like(xyz)

    elements = np.empty(xyz.shape[0], dtype=dtype)
    attributes = np.concatenate((xyz, normals, rgb), axis=1)
    elements[:] = list(map(tuple, attributes))

    # Create the PlyData object and write to file
    vertex_element = PlyElement.describe(elements, 'vertex')
    ply_data = PlyData([vertex_element])
    ply_data.write(path)
