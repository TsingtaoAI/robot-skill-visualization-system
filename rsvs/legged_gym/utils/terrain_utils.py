import numpy as np
from scipy import interpolate
import trimesh
from typing import Tuple, List

class SubTerrain:
    def __init__(self, 
                 terrain_name : str ="terrain", 
                 width : int =256, 
                 length: int =256, 
                 vertical_scale: float =1.0, 
                 horizontal_scale: float =1.0):
        """
        Initialize a SubTerrain object.

        Args:
            terrain_name (str, optional): Name of the terrain. Defaults to "terrain".
            width (int, optional): Width of the terrain (number of pixels). Defaults to 256.
            length (int, optional): Length of the terrain (number of pixels). Defaults to 256.
            vertical_scale (float, optional): Vertical scale of the terrain. Defaults to 1.0.
            horizontal_scale (float, optional): Horizontal scale of the terrain. Defaults to 1.0.
        """
        self.terrain_name = terrain_name
        self.vertical_scale = vertical_scale
        self.horizontal_scale = horizontal_scale
        self.width = width
        self.length = length
        self.height_field_raw = np.zeros((self.length, self.width), dtype=np.int16)
        # add a trimesh object to store the trimesh of subterrain, for use in trimesh terrain type
        self.terrain_mesh = trimesh.Trimesh()
        # add edge mask to indicate the edge points of the terrain, for use in rewards
        self.edge_mask = np.zeros((self.length, self.width), dtype=bool)

#---------- Heightfield Terrain Functions ----------#

def random_uniform_terrain(terrain : SubTerrain, 
                           min_height: float, 
                           max_height: float, 
                           step: float = 1, 
                           downsampled_scale: float = None,
                           terrain_type: str = None) -> SubTerrain:
    """
    Generate a uniform noise terrain

    Parameters:
        terrain (SubTerrain): the terrain
        min_height (float): the minimum height of the terrain [meters]
        max_height (float): the maximum height of the terrain [meters]
        step (float): minimum height change between two points [meters]
        downsampled_scale (float): distance between two randomly sampled points ( musty be larger or equal to terrain.horizontal_scale)

    Note:
        This function can only generate terrain from heightfield
    
    """
    if downsampled_scale is None:
        downsampled_scale = terrain.horizontal_scale
    if terrain_type in [None, "plane"]:
        raise ValueError("random_uniform_terrain can only be used for heightfield or trimesh terrain type")

    flat_edge = int(0.2 / terrain.horizontal_scale) # 20cm flat edge around the terrain
    # switch parameters to discrete units
    min_height = int(min_height / terrain.vertical_scale)
    max_height = int(max_height / terrain.vertical_scale)
    step = int(step / terrain.vertical_scale)

    heights_range = np.arange(min_height, max_height + step, step)
    height_field_downsampled = np.random.choice(heights_range, 
                                                (int((terrain.length - 2 * flat_edge) * terrain.horizontal_scale / downsampled_scale), 
                                                 int((terrain.width - 2 * flat_edge) * terrain.horizontal_scale / downsampled_scale)))

    x = np.linspace(flat_edge * terrain.horizontal_scale, 
                    (terrain.length - flat_edge) * terrain.horizontal_scale, height_field_downsampled.shape[0])
    y = np.linspace(flat_edge * terrain.horizontal_scale, 
                    (terrain.width - flat_edge) * terrain.horizontal_scale, height_field_downsampled.shape[1])

    # f = interpolate.interp2d(y, x, height_field_downsampled, kind='linear')
    f = interpolate.RectBivariateSpline(
        y, x, height_field_downsampled, kx=1, ky=1
    )

    x_upsampled = np.linspace(flat_edge * terrain.horizontal_scale, 
                              (terrain.length - flat_edge) * terrain.horizontal_scale, 
                              terrain.length - 2 * flat_edge)
    y_upsampled = np.linspace(flat_edge * terrain.horizontal_scale, 
                              (terrain.width - flat_edge) * terrain.horizontal_scale, 
                              terrain.width - 2 * flat_edge)
    z_upsampled = np.rint(f(y_upsampled, x_upsampled))

    terrain.height_field_raw[flat_edge:-flat_edge, flat_edge:-flat_edge] += z_upsampled.astype(np.int16)
    
    # generate the terrain mesh for trimesh terrain type
    if terrain_type == "trimesh":
        vertices, triangles = convert_heightfield_to_trimesh(terrain.height_field_raw, terrain.horizontal_scale, terrain.vertical_scale)
        terrain_mesh = trimesh.Trimesh(vertices=vertices, faces=triangles)
        # add a border mesh to avoid holes at the edges of the terrain
        border_meshes = make_border(
            size=(terrain.length * terrain.horizontal_scale, 
                  terrain.width * terrain.horizontal_scale),
            inner_size=((terrain.length - 2) * terrain.horizontal_scale, 
                        (terrain.width - 2) * terrain.horizontal_scale),
            height=1.0,
            position=(0.5 * terrain.length * terrain.horizontal_scale, 
                      0.5 * terrain.width * terrain.horizontal_scale, 
                      -0.5)
        )
        border_mesh = trimesh.util.concatenate(border_meshes)
        # update the faces to have minimal triangles
        selector = ~(np.asarray(border_mesh.triangles)[:, :, 2] < -0.1).any(1)
        border_mesh.update_faces(selector)
        # add a small offset to align the terrain mesh with the border
        translation = np.array([
                terrain.horizontal_scale,
                terrain.horizontal_scale,
                0
            ])
        terrain_mesh.apply_translation(translation)
        terrain.terrain_mesh = trimesh.util.concatenate([terrain_mesh, border_mesh])
    
    return terrain

def pyramid_sloped_terrain(terrain: SubTerrain, 
                           slope: float = 1, 
                           platform_size: float = 1., 
                           terrain_type: str = None) -> SubTerrain:
    """
    Generate a sloped terrain

    Parameters:
        terrain (terrain): the terrain
        slope (int): positive or negative slope
        platform_size (float): size of the flat platform at the center of the terrain [meters]
    Returns:
        terrain (SubTerrain): update terrain
    Note:
        This function can only generate terrain from heightfield
    """
    if terrain_type in [None, "plane"]:
        raise ValueError("pyramid_sloped_terrain can only be used for heightfield or trimesh terrain type")

    flat_edge = int(0.2 / terrain.horizontal_scale) # 20cm flat edge around the terrain
    
    x = np.arange(flat_edge, terrain.length - flat_edge)
    y = np.arange(flat_edge, terrain.width - flat_edge)
    center_x = int(terrain.length / 2)
    center_y = int(terrain.width / 2)
    xx, yy = np.meshgrid(x, y, sparse=True)
    xx = (center_x - np.abs(center_x-xx)) / center_x
    yy = (center_y - np.abs(center_y-yy)) / center_y
    xx = xx.reshape(terrain.length - 2 * flat_edge, 1)
    yy = yy.reshape(1, terrain.width - 2 * flat_edge)
    max_height = int(slope * (terrain.horizontal_scale / terrain.vertical_scale) * ((terrain.length - 2 * flat_edge) / 2))
    terrain.height_field_raw[flat_edge:-flat_edge, flat_edge:-flat_edge] += (max_height * xx * yy).astype(terrain.height_field_raw.dtype)

    platform_size = int(platform_size / terrain.horizontal_scale / 2)
    x1 = terrain.length // 2 - platform_size
    x2 = terrain.length // 2 + platform_size
    y1 = terrain.width // 2 - platform_size
    y2 = terrain.width // 2 + platform_size

    min_h = min(terrain.height_field_raw[x1, y1], 0)
    max_h = max(terrain.height_field_raw[x1, y1], 0)
    terrain.height_field_raw = np.clip(terrain.height_field_raw, min_h, max_h)
    
    # generate the terrain mesh for trimesh terrain type
    if terrain_type == "trimesh":
        vertices, triangles = convert_heightfield_to_trimesh(terrain.height_field_raw, terrain.horizontal_scale, terrain.vertical_scale)
        terrain_mesh = trimesh.Trimesh(vertices=vertices, faces=triangles)
        # add a border mesh to avoid holes at the edges of the terrain
        border_meshes = make_border(
            size=(terrain.length * terrain.horizontal_scale,
                  terrain.width * terrain.horizontal_scale),
            inner_size=((terrain.length - 2) * terrain.horizontal_scale,
                        (terrain.width - 2) * terrain.horizontal_scale),
            height=1.0,
            position=(0.5 * terrain.length * terrain.horizontal_scale, 
                      0.5 * terrain.width * terrain.horizontal_scale, 
                      -0.5)
        )
        border_mesh = trimesh.util.concatenate(border_meshes)
        # update the faces to have minimal triangles
        selector = ~(np.asarray(border_mesh.triangles)[:, :, 2] < -0.1).any(1)
        border_mesh.update_faces(selector)
        # add a small offset to align the terrain mesh with the border
        translation = np.array([
                terrain.horizontal_scale,
                terrain.horizontal_scale,
                0
            ])
        terrain_mesh.apply_translation(translation)
        terrain.terrain_mesh = trimesh.util.concatenate([terrain_mesh, border_mesh])
    
    return terrain


def discrete_obstacles_terrain(terrain : SubTerrain, 
                               max_height : float, 
                               min_size : float, 
                               max_size : float, 
                               num_rects : int, 
                               platform_size: float = 1.,
                               terrain_type: str = None,
                               simplify_mesh: bool = False) -> SubTerrain:
    """
    Generate a terrain with gaps

    Parameters:
        terrain (SubTerrain): the terrain
        max_height (float): maximum height of the obstacles (range=[-max, -max/2, max/2, max]) [meters]
        min_size (float): minimum size of a rectangle obstacle [meters]
        max_size (float): maximum size of a rectangle obstacle [meters]
        num_rects (int): number of randomly generated obstacles
        platform_size (float): size of the flat platform at the center of the terrain [meters]
        terrain_type (str): type of the terrain ("heightfield" or "trimesh")
        simplify_mesh (bool): whether to simplify the generated mesh
    Returns:
        terrain (SubTerrain): update terrain
    """
    if terrain_type in [None, "plane"]:
        raise ValueError("discrete_obstacles_terrain can only be used for heightfield or trimesh terrain type")
    
    # switch parameters to discrete units
    max_height = int(max_height / terrain.vertical_scale)
    min_size = int(min_size / terrain.horizontal_scale)
    max_size = int(max_size / terrain.horizontal_scale)
    platform_size = int(platform_size / terrain.horizontal_scale)

    (i, j) = terrain.height_field_raw.shape
    height_range = [max_height // 4, max_height // 2, max_height // 4 * 3, max_height]
    width_range = range(min_size, max_size, 4)
    length_range = range(min_size, max_size, 4)
    
    platform_x1 = (terrain.length - platform_size) // 2
    platform_x2 = (terrain.length + platform_size) // 2
    platform_y1 = (terrain.width - platform_size) // 2
    platform_y2 = (terrain.width + platform_size) // 2

    if terrain_type == "trimesh":
        # generate terrain mesh along with the heightfield simultaneously
        # initialize list of meshes
        meshes_list = list()

    placed_rects = []

    for _ in range(num_rects):
        width = np.random.choice(width_range)
        length = np.random.choice(length_range)

        valid_starts = []
        for start_i in range(0, i - length, 4):
            for start_j in range(0, j - width, 4):
                overlap_x = start_i < platform_x2 and (start_i + length) > platform_x1
                overlap_y = start_j < platform_y2 and (start_j + width) > platform_y1
                # Reject only when the obstacle intersects platform on both axes.
                if overlap_x and overlap_y:
                    continue

                # Reject candidates that intersect any existing rectangle.
                intersects_existing = False
                for r_x1, r_x2, r_y1, r_y2 in placed_rects:
                    overlaps_existing_x = start_i < r_x2 and (start_i + length) > r_x1
                    overlaps_existing_y = start_j < r_y2 and (start_j + width) > r_y1
                    if overlaps_existing_x and overlaps_existing_y:
                        intersects_existing = True
                        break

                if not intersects_existing:
                    valid_starts.append((start_i, start_j))

        if len(valid_starts) == 0:
            continue

        start_i, start_j = valid_starts[np.random.randint(len(valid_starts))]
        placed_rects.append((start_i, start_i + length, start_j, start_j + width))
        height = np.random.choice(height_range)
        terrain.height_field_raw[start_i:start_i+length, start_j:start_j+width] = height
        if terrain_type == "trimesh":
            box_center = ((start_i + length / 2) * terrain.horizontal_scale, 
                          (start_j + width / 2) * terrain.horizontal_scale, 
                          height * terrain.vertical_scale / 2)
            box_dim = (length * terrain.horizontal_scale, 
                       width * terrain.horizontal_scale, 
                       abs(height) * terrain.vertical_scale)
            box_mesh = trimesh.creation.box(box_dim, trimesh.transformations.translation_matrix(box_center))
            meshes_list.append(box_mesh)

    x1 = platform_x1
    x2 = platform_x2
    y1 = platform_y1
    y2 = platform_y2
    terrain.height_field_raw[x1:x2, y1:y2] = 0
    
    # Identify the edge points of the terrain and set the edge mask accordingly
    edge_threshold = int(0.04 / terrain.vertical_scale) # 4cm height difference to identify edge points
    # if the height difference between a point and its 4-connected neighbors is greater than the threshold, then this point is an edge point
    for i in range(0, terrain.length - 1):
        for j in range(0, terrain.width - 1):
            if (abs(terrain.height_field_raw[i, j] - terrain.height_field_raw[i-1, j]) > edge_threshold or
                abs(terrain.height_field_raw[i, j] - terrain.height_field_raw[i+1, j]) > edge_threshold or
                abs(terrain.height_field_raw[i, j] - terrain.height_field_raw[i, j-1]) > edge_threshold or
                abs(terrain.height_field_raw[i, j] - terrain.height_field_raw[i, j+1]) > edge_threshold):
                terrain.edge_mask[i, j] = True
    
    # generate the terrain mesh for trimesh terrain type
    if terrain_type == "trimesh":
        if not simplify_mesh:
          vertices, triangles = convert_heightfield_to_trimesh(terrain.height_field_raw, terrain.horizontal_scale, terrain.vertical_scale)
          terrain_mesh = trimesh.Trimesh(vertices=vertices, faces=triangles)
          # add a border mesh to avoid holes at the edges of the terrain
          border_meshes = make_border(
              size=(terrain.length * terrain.horizontal_scale,
                    terrain.width * terrain.horizontal_scale),
              inner_size=((terrain.length - 2) * terrain.horizontal_scale,
                          (terrain.width - 2) * terrain.horizontal_scale),
              height=1.0,
              position=(0.5 * terrain.length * terrain.horizontal_scale, 
                        0.5 * terrain.width * terrain.horizontal_scale, 
                        -0.5)
          )
          border_mesh = trimesh.util.concatenate(border_meshes)
          # update the faces to have minimal triangles
          selector = ~(np.asarray(border_mesh.triangles)[:, :, 2] < -0.1).any(1)
          border_mesh.update_faces(selector)
          # add a small offset to align the terrain mesh with the border
          translation = np.array([
                  terrain.horizontal_scale,
                  terrain.horizontal_scale,
                  0
              ])
          terrain_mesh.apply_translation(translation)
          terrain.terrain_mesh = trimesh.util.concatenate([terrain_mesh, border_mesh])
        
        else:
          # add ground mesh
          ground_center = (terrain.length * terrain.horizontal_scale / 2,
                          terrain.width * terrain.horizontal_scale / 2,
                          -0.5)
          ground_dim = (terrain.length * terrain.horizontal_scale,
                        terrain.width * terrain.horizontal_scale,
                        1.0)
          ground_mesh = trimesh.creation.box(ground_dim, trimesh.transformations.translation_matrix(ground_center))
          meshes_list.append(ground_mesh)
          terrain.terrain_mesh = trimesh.util.concatenate(meshes_list)
    
    return terrain


def wave_terrain(terrain : SubTerrain, 
                 num_waves: int = 1, 
                 amplitude: float = 1.,
                 terrain_type: str = None) -> SubTerrain:
    """
    Generate a wavy terrain

    Parameters:
        terrain (SubTerrain): the terrain
        num_waves (int): number of sine waves across the terrain length
        amplitude (float): amplitude of the waves
        terrain_type (str): type of the terrain ("heightfield" or "trimesh")
    Returns:
        terrain (SubTerrain): update terrain
    Note:
        This function can only generate terrain from heightfield
    """
    if terrain_type in [None, "plane"]:
        raise ValueError("wave_terrain can only be used for heightfield or trimesh terrain type")
    
    amplitude = int(0.5*amplitude / terrain.vertical_scale)
    flat_edge = int(0.2 / terrain.horizontal_scale) # 20cm flat edge around the terrain
    if num_waves > 0:
        div = terrain.length / (num_waves * np.pi * 2)
        x = np.arange(flat_edge, terrain.length - flat_edge)
        y = np.arange(flat_edge, terrain.width - flat_edge)
        xx, yy = np.meshgrid(x, y, sparse=True)
        xx = xx.reshape(terrain.length - 2*flat_edge, 1)
        yy = yy.reshape(1, terrain.width - 2*flat_edge)
        terrain.height_field_raw[flat_edge:terrain.length-flat_edge, flat_edge:terrain.width-flat_edge] += (amplitude*np.cos(yy / div) + amplitude*np.sin(xx / div)).astype(
            terrain.height_field_raw.dtype)
    
    # generate the terrain mesh for trimesh terrain type
    if terrain_type == "trimesh":
        vertices, triangles = convert_heightfield_to_trimesh(terrain.height_field_raw, terrain.horizontal_scale, terrain.vertical_scale)
        terrain_mesh = trimesh.Trimesh(vertices=vertices, faces=triangles)
        # add a border mesh to avoid holes at the edges of the terrain
        border_meshes = make_border(
            size=(terrain.length * terrain.horizontal_scale,
                  terrain.width * terrain.horizontal_scale),
            inner_size=((terrain.length - 2) * terrain.horizontal_scale,
                        (terrain.width - 2) * terrain.horizontal_scale),
            height=1.0,
            position=(0.5 * terrain.length * terrain.horizontal_scale, 
                      0.5 * terrain.width * terrain.horizontal_scale, 
                      -0.5)
        )
        border_mesh = trimesh.util.concatenate(border_meshes)
        # update the faces to have minimal triangles
        selector = ~(np.asarray(border_mesh.triangles)[:, :, 2] < -0.1).any(1)
        border_mesh.update_faces(selector)
        # add a small offset to align the terrain mesh with the border
        translation = np.array([
                terrain.horizontal_scale,
                terrain.horizontal_scale,
                0
            ])
        terrain_mesh.apply_translation(translation)
        terrain.terrain_mesh = trimesh.util.concatenate([terrain_mesh, border_mesh])
        
    return terrain

def pyramid_stairs_terrain(terrain : SubTerrain, 
                           step_width : float, 
                           step_height : float, 
                           platform_size: float = 1.,
                           terrain_type: str = None,
                           simplify_mesh: bool = False) -> SubTerrain:
    """
    Generate stairs

    Parameters:
        terrain (SubTerrain): the terrain
        step_width (float):  the width of the step [meters]
        step_height (float): the step_height [meters]
        platform_size (float): size of the flat platform at the center of the terrain [meters]
        terrain_type (str): type of the terrain ("heightfield" or "trimesh")
        simplify_mesh (bool): whether to simplify the generated mesh
    Returns:
        terrain (SubTerrain): update terrain
    Note:
        This function will use mesh_pyramid_stairs_terrain to directly generate the mesh
    """
    if terrain_type in [None, "plane"]:
        raise ValueError("pyramid_stairs_terrain can only be used for heightfield or trimesh terrain type")
    
    # switch parameters to discrete units
    step_width = int(step_width / terrain.horizontal_scale)
    step_height = int(step_height / terrain.vertical_scale)
    platform_size = int(platform_size / terrain.horizontal_scale)

    height = 0
    start_x = 0
    stop_x = terrain.length
    start_y = 0
    stop_y = terrain.width
    while (stop_x - start_x) > platform_size and (stop_y - start_y) > platform_size:
        terrain.height_field_raw[start_x: stop_x, start_y: stop_y] = height
        start_x += step_width
        stop_x -= step_width
        start_y += step_width
        stop_y -= step_width
        height += step_height
    
    # Identify the edge points of the terrain and set the edge mask accordingly
    edge_threshold = int(0.04 / terrain.vertical_scale) # 4cm height difference to identify edge points
    # if the height difference between a point and its 4-connected neighbors is greater than the threshold, then this point is an edge point
    for i in range(0, terrain.length - 1):
        for j in range(0, terrain.width - 1):
            if (abs(terrain.height_field_raw[i, j] - terrain.height_field_raw[i-1, j]) > edge_threshold or
                abs(terrain.height_field_raw[i, j] - terrain.height_field_raw[i+1, j]) > edge_threshold or
                abs(terrain.height_field_raw[i, j] - terrain.height_field_raw[i, j-1]) > edge_threshold or
                abs(terrain.height_field_raw[i, j] - terrain.height_field_raw[i, j+1]) > edge_threshold):
                terrain.edge_mask[i, j] = True
    
    # generate the terrain mesh for trimesh terrain type
    if terrain_type == "trimesh":
        if not simplify_mesh:
          vertices, triangles = convert_heightfield_to_trimesh(terrain.height_field_raw, terrain.horizontal_scale, terrain.vertical_scale)
          terrain.terrain_mesh = trimesh.Trimesh(vertices=vertices, faces=triangles)
          # add a small offset to align the terrain mesh with the border
          translation = np.array([
                  terrain.horizontal_scale / 2.0,
                  terrain.horizontal_scale / 2.0,
                  0
              ])
          terrain.terrain_mesh.apply_translation(translation)
        
        else:
          if step_height >= 0:
              terrain.terrain_mesh = mesh_pyramid_stairs_terrain(terrain,
                                                            step_width=step_width * terrain.horizontal_scale, 
                                                            step_height=step_height * terrain.vertical_scale, 
                                                            platform_size=platform_size * terrain.horizontal_scale)
          elif step_height < 0:
              terrain.terrain_mesh = mesh_inverted_pyramid_stairs_terrain(terrain,
                                                            step_width=step_width * terrain.horizontal_scale, 
                                                            step_height=-step_height * terrain.vertical_scale, 
                                                            platform_size=platform_size * terrain.horizontal_scale)
        
    return terrain


def stepping_stones_terrain(terrain : SubTerrain, 
                            stone_length : float,
                            stone_width : float, 
                            stone_distance_x : float,
                            stone_distance_y : float, 
                            max_height : float, 
                            platform_size : float =1., 
                            depth : float =-10,
                            terrain_type: str = None,
                            simplify_mesh: bool = False) -> SubTerrain:
    """
    Generate a stepping stones terrain

    Parameters:
        terrain (SubTerrain): the terrain
        stone_length (float): length of the stepping stones (X direction) [meters]
        stone_width (float): width of the stepping stones (Y direction) [meters]
        stone_distance_x (float): distance between stones in x-direction (i.e size of the holes) [meters]
        stone_distance_y (float): distance between stones in y-direction (i.e size of the holes) [meters]
        max_height (float): maximum height of the stones (positive and negative) [meters]
        platform_size (float): size of the flat platform at the center of the terrain [meters]
        depth (float): depth of the holes (default=-10.) [meters]
        terrain_type (str): type of the terrain (heightfield or trimesh)
        simplify_mesh (bool): whether to simplify the mesh
    Returns:
        terrain (SubTerrain): update terrain
    """
    if terrain_type in [None, "plane"]:
        raise ValueError("stepping_stones_terrain can only be used for heightfield or trimesh terrain type")
    
    # switch parameters to discrete units
    stone_length = int(stone_length / terrain.horizontal_scale)
    stone_width = int(stone_width / terrain.horizontal_scale)
    stone_distance_x = int(stone_distance_x / terrain.horizontal_scale)
    stone_distance_y = int(stone_distance_y / terrain.horizontal_scale)
    max_height = int(max_height / terrain.vertical_scale)
    platform_size = int(platform_size / terrain.horizontal_scale)
    height_range = np.arange(-max_height-1, max_height, step=1)
    
    if terrain_type == "trimesh":
        # generate terrain mesh along with the heightfield simultaneously
        # initialize list of meshes
        meshes_list = list()
        
    start_x = 0
    start_y = 0
    terrain.height_field_raw[:, :] = int(depth / terrain.vertical_scale)
    if terrain.length >= terrain.width:
        while start_y < terrain.width:
            stop_y = min(terrain.width, start_y + stone_width)
            start_x = 0
            # fill row
            while start_x < terrain.length:
                stop_x = min(terrain.length, start_x + stone_length)
                height = np.random.choice(height_range)
                terrain.height_field_raw[start_x: stop_x, start_y: stop_y] = height
                # generate the box mesh for the stone
                if terrain_type == "trimesh":
                    box_center = ((stop_x + start_x) / 2 * terrain.horizontal_scale, 
                                  (start_y + stop_y) / 2 * terrain.horizontal_scale,
                                  depth / 2)
                    box_dim = ((stop_x - start_x) * terrain.horizontal_scale, 
                               (stop_y - start_y) * terrain.horizontal_scale,
                               abs(depth))
                    box_mesh = trimesh.creation.box(box_dim, trimesh.transformations.translation_matrix(box_center))
                    meshes_list.append(box_mesh)
                    
                start_x += stone_length + stone_distance_x
            start_y += stone_width + stone_distance_y
    elif terrain.width > terrain.length:
        while start_x < terrain.length:
            stop_x = min(terrain.length, start_x + stone_length)
            start_y = 0
            # fill column
            while start_y < terrain.width:
                stop_y = min(terrain.width, start_y + stone_width)
                height = np.random.choice(height_range)
                terrain.height_field_raw[start_x: stop_x, start_y: stop_y] = height
                # generate the box mesh for the stone
                if terrain_type == "trimesh":
                    box_center = ((stop_x + start_x) / 2 * terrain.horizontal_scale, 
                                  (start_y + stop_y) / 2 * terrain.horizontal_scale,
                                    depth)
                    box_dim = ((stop_x - start_x) * terrain.horizontal_scale, 
                               (stop_y - start_y) * terrain.horizontal_scale,
                               abs(depth))
                    box_mesh = trimesh.creation.box(box_dim, trimesh.transformations.translation_matrix(box_center))
                    meshes_list.append(box_mesh)
                    
                start_y += stone_width + stone_distance_y
            start_x += stone_length + stone_distance_x

    x1 = (terrain.length - platform_size) // 2
    x2 = (terrain.length + platform_size) // 2
    y1 = (terrain.width - platform_size) // 2
    y2 = (terrain.width + platform_size) // 2
    terrain.height_field_raw[x1:x2, y1:y2] = 0
    # generate the platform mesh for the center flat area
    if terrain_type == "trimesh":
        platform_center = (0.5 * (x1 + x2) * terrain.horizontal_scale, 
                           0.5 * (y1 + y2) * terrain.horizontal_scale, 
                           -0.05)
        platform_dim = (platform_size * terrain.horizontal_scale, 
                        platform_size * terrain.horizontal_scale, 
                        0.1)
        platform_mesh = trimesh.creation.box(platform_dim, trimesh.transformations.translation_matrix(platform_center))
        meshes_list.append(platform_mesh)
    
    # Identify the edge points of the terrain and set the edge mask accordingly
    edge_threshold = int(0.04 / terrain.vertical_scale) # 4cm height difference to identify edge points
    # if the height difference between a point and its 4-connected neighbors is greater than the threshold, then this point is an edge point
    for i in range(0, terrain.length - 1):
        for j in range(0, terrain.width - 1):
            if (abs(terrain.height_field_raw[i, j] - terrain.height_field_raw[i-1, j]) > edge_threshold or
                abs(terrain.height_field_raw[i, j] - terrain.height_field_raw[i+1, j]) > edge_threshold or
                abs(terrain.height_field_raw[i, j] - terrain.height_field_raw[i, j-1]) > edge_threshold or
                abs(terrain.height_field_raw[i, j] - terrain.height_field_raw[i, j+1]) > edge_threshold):
                terrain.edge_mask[i, j] = True
    
    # generate the terrain mesh for trimesh terrain type
    if terrain_type == "trimesh":
      if not simplify_mesh:
        vertices, triangles = convert_heightfield_to_trimesh(terrain.height_field_raw, terrain.horizontal_scale, terrain.vertical_scale)
        terrain.terrain_mesh = trimesh.Trimesh(vertices=vertices, faces=triangles)
        # add a small offset to align the terrain mesh with the border
        translation = np.array([
                terrain.horizontal_scale / 2.0,
                terrain.horizontal_scale / 2.0,
                0
            ])
        terrain.terrain_mesh.apply_translation(translation)
      
      else:
        # generate the bottom mesh for the holes
        bottom_center = (terrain.length * terrain.horizontal_scale / 2, 
                         terrain.width * terrain.horizontal_scale / 2, 
                         depth - 0.05)
        bottom_dim = (terrain.length * terrain.horizontal_scale, 
                     terrain.width * terrain.horizontal_scale,
                     0.1)
        bottom_mesh = trimesh.creation.box(bottom_dim, trimesh.transformations.translation_matrix(bottom_center))
        meshes_list.append(bottom_mesh)
        
        # concatenate all the stone meshes and the platform mesh to form the terrain mesh
        terrain.terrain_mesh = trimesh.util.concatenate(meshes_list)
    
    return terrain

def gap_terrain(terrain : SubTerrain, 
                gap_size : float, 
                platform_size : float =1.,
                terrain_type : str = None,
                simplify_mesh : bool = False) -> SubTerrain:
    if terrain_type in [None, "plane"]:
        raise ValueError("gap_terrain can only be used for heightfield or trimesh terrain type")
    
    gap_size = int(gap_size / terrain.horizontal_scale)
    platform_size = int(platform_size / terrain.horizontal_scale)

    center_x = terrain.length // 2
    center_y = terrain.width // 2
    x1 = platform_size // 2
    x2 = x1 + gap_size
    y1 = platform_size // 2
    y2 = y1 + gap_size
   
    terrain.height_field_raw[center_x-x2 : center_x + x2, center_y-y2 : center_y + y2] = -1000
    terrain.height_field_raw[center_x-x1 : center_x + x1, center_y-y1 : center_y + y1] = 0

    # Identify the edge points of the terrain and set the edge mask accordingly
    edge_threshold = int(0.04 / terrain.vertical_scale) # 4cm height difference to identify edge points
    # if the height difference between a point and its 4-connected neighbors is greater than the threshold, then this point is an edge point
    for i in range(0, terrain.length - 1):
        for j in range(0, terrain.width - 1):
            if (abs(terrain.height_field_raw[i, j] - terrain.height_field_raw[i-1, j]) > edge_threshold or
                abs(terrain.height_field_raw[i, j] - terrain.height_field_raw[i+1, j]) > edge_threshold or
                abs(terrain.height_field_raw[i, j] - terrain.height_field_raw[i, j-1]) > edge_threshold or
                abs(terrain.height_field_raw[i, j] - terrain.height_field_raw[i, j+1]) > edge_threshold):
                terrain.edge_mask[i, j] = True
    
    # generate the terrain mesh for trimesh terrain type
    if terrain_type == "trimesh":
      
      if not simplify_mesh:
        vertices, triangles = convert_heightfield_to_trimesh(terrain.height_field_raw, terrain.horizontal_scale, terrain.vertical_scale)
        terrain.terrain_mesh = trimesh.Trimesh(vertices=vertices, faces=triangles)
        # add a small offset to align the terrain mesh with the border
        translation = np.array([
                terrain.horizontal_scale / 2.0,
                terrain.horizontal_scale / 2.0,
                0
            ])
        terrain.terrain_mesh.apply_translation(translation)
      
      else:
        terrain.terrain_mesh = mesh_gap_terrain(terrain,
                                               gap_size=gap_size * terrain.horizontal_scale, 
                                               platform_size=platform_size * terrain.horizontal_scale)
    
    return terrain
    
def pit_terrain(terrain : SubTerrain, 
                depth : float, 
                platform_size : float =1.,
                terrain_type : str = None,
                simplify_mesh : bool = False) -> SubTerrain:
    depth = int(depth / terrain.vertical_scale)
    platform_size = int(platform_size / terrain.horizontal_scale / 2) # half platform size for easier calculation
    x1 = terrain.length // 2 - platform_size
    x2 = terrain.length // 2 + platform_size
    y1 = terrain.width // 2 - platform_size
    y2 = terrain.width // 2 + platform_size
    terrain.height_field_raw[x1:x2, y1:y2] = -depth
    
    # Identify the edge points of the terrain and set the edge mask accordingly
    edge_threshold = int(0.04 / terrain.vertical_scale) # 4cm height difference to identify edge points
    # if the height difference between a point and its 4-connected neighbors is greater than the threshold, then this point is an edge point
    for i in range(0, terrain.length - 1):
        for j in range(0, terrain.width - 1):
            if (abs(terrain.height_field_raw[i, j] - terrain.height_field_raw[i-1, j]) > edge_threshold or
                abs(terrain.height_field_raw[i, j] - terrain.height_field_raw[i+1, j]) > edge_threshold or
                abs(terrain.height_field_raw[i, j] - terrain.height_field_raw[i, j-1]) > edge_threshold or
                abs(terrain.height_field_raw[i, j] - terrain.height_field_raw[i, j+1]) > edge_threshold):
                terrain.edge_mask[i, j] = True
    
    # generate the terrain mesh for trimesh terrain type
    if terrain_type == "trimesh":
      if not simplify_mesh:
        vertices, triangles = convert_heightfield_to_trimesh(terrain.height_field_raw, terrain.horizontal_scale, terrain.vertical_scale)
        terrain.terrain_mesh = trimesh.Trimesh(vertices=vertices, faces=triangles)
        # add a small offset to align the terrain mesh with the border
        translation = np.array([
                terrain.horizontal_scale / 2.0,
                terrain.horizontal_scale / 2.0,
                0
            ])
        terrain.terrain_mesh.apply_translation(translation)
      
      else:
        # platform_size in heightfield is half-width (radius) in cells
        # mesh_pit_terrain expects full platform size in meters
        terrain.terrain_mesh = mesh_pit_terrain(terrain, 
                                               depth=depth * terrain.vertical_scale, 
                                               platform_size=platform_size * 2 * terrain.horizontal_scale)
    
    return terrain

def multiple_high_platforms_terrain(terrain : SubTerrain,
                          high_platform_height: float,
                          high_platform_length: float,
                          high_platform_width: float,
                          high_platform_interval: float,
                          platform_size: float = 1.,
                          terrain_type: str = None,
                          simplify_mesh: bool = False) -> SubTerrain:
    """Generate multiple high platforms in the X direction track

    Args:
        terrain (SubTerrain): subterrain object to be updated
        high_platform_height (float): the height of the high platform (positive value, in meters)
        high_platform_length (float): the length of the high platform in X direction (in meters)
        high_platform_width (float): the width of the high platform in Y direction (in meters)
        high_platform_interval (float): the interval between two high platforms (in meters)
        platform_size (float, optional): size of the platform in the middle of the terrain. Defaults to 1..
        terrain_type (str, optional): type of the terrain. Defaults to None.
        simplify_mesh (bool, optional): whether to simplify the terrain mesh. Defaults to False.
    Returns:
        SubTerrain: updated terrain with multiple high platforms
    """
    if terrain_type in [None, "plane"]:
        raise ValueError("gap_terrain can only be used for heightfield or trimesh terrain type")
    
    # convert values from meters to discrete units
    high_platform_height = int(high_platform_height / terrain.vertical_scale)
    high_platform_length = int(high_platform_length / terrain.horizontal_scale)
    high_platform_width = int(high_platform_width / terrain.horizontal_scale)
    high_platform_interval = int(high_platform_interval / terrain.horizontal_scale)
    platform_size = int(platform_size / terrain.horizontal_scale / 2)
    
    if terrain_type == "trimesh":
        # generate terrain mesh along with the heightfield simultaneously
        # initialize list of meshes
        meshes_list = list()

    # positive x direction
    x_center = terrain.length // 2
    y_center = terrain.width // 2
    y_start = y_center -high_platform_width // 2
    y_end = y_center + high_platform_width // 2
    x_start = x_center + platform_size
    x_end = x_start + high_platform_length
    while x_end < terrain.length:
        terrain.height_field_raw[x_start:x_end, y_start:y_end] = high_platform_height
        if terrain_type == "trimesh":
            box_center = (0.5 * (x_start + x_end) * terrain.horizontal_scale,
                          0.5 * (y_start + y_end) * terrain.horizontal_scale,
                          0.5 * high_platform_height * terrain.vertical_scale)
            box_dim = ((x_end - x_start) * terrain.horizontal_scale,
                       (y_end - y_start) * terrain.horizontal_scale,
                       high_platform_height * terrain.vertical_scale)
            box_mesh = trimesh.creation.box(box_dim, trimesh.transformations.translation_matrix(box_center))
            meshes_list.append(box_mesh)
        x_start = x_end + high_platform_interval
        x_end = x_start + high_platform_length
    
    # negative x direction
    x_end = x_center - platform_size
    x_start = x_end - high_platform_length
    while x_start > 0:
        terrain.height_field_raw[x_start:x_end, y_start:y_end] = high_platform_height
        if terrain_type == "trimesh":
            box_center = (0.5 * (x_start + x_end) * terrain.horizontal_scale,
                          0.5 * (y_start + y_end) * terrain.horizontal_scale,
                          0.5 * high_platform_height * terrain.vertical_scale)
            box_dim = ((x_end - x_start) * terrain.horizontal_scale,
                       (y_end - y_start) * terrain.horizontal_scale,
                       high_platform_height * terrain.vertical_scale)
            box_mesh = trimesh.creation.box(box_dim, trimesh.transformations.translation_matrix(box_center))
            meshes_list.append(box_mesh)
        x_end = x_start - high_platform_interval
        x_start = x_end - high_platform_length
    
    # Identify the edge points of the terrain and set the edge mask accordingly
    edge_threshold = int(0.04 / terrain.vertical_scale) # 4cm height difference to identify edge points
    # if the height difference between a point and its 4-connected neighbors is greater than the threshold, then this point is an edge point
    for i in range(0, terrain.length - 1):
        for j in range(0, terrain.width - 1):
            if (abs(terrain.height_field_raw[i, j] - terrain.height_field_raw[i-1, j]) > edge_threshold or
                abs(terrain.height_field_raw[i, j] - terrain.height_field_raw[i+1, j]) > edge_threshold or
                abs(terrain.height_field_raw[i, j] - terrain.height_field_raw[i, j-1]) > edge_threshold or
                abs(terrain.height_field_raw[i, j] - terrain.height_field_raw[i, j+1]) > edge_threshold):
                terrain.edge_mask[i, j] = True
    
    # add a flat ground mesh for the rest of the terrain and concatenate with the high_platform meshes to get the final terrain mesh
    if terrain_type == "trimesh":
      if not simplify_mesh:
        vertices, triangles = convert_heightfield_to_trimesh(terrain.height_field_raw, terrain.horizontal_scale, terrain.vertical_scale)
        terrain.terrain_mesh = trimesh.Trimesh(vertices=vertices, faces=triangles)
        # add a small offset to align the terrain mesh with the border
        translation = np.array([
                terrain.horizontal_scale / 2.0,
                terrain.horizontal_scale / 2.0,
                0
            ])
        terrain.terrain_mesh.apply_translation(translation)
        
      else:
        # add ground mesh
        ground_center = (0.5 * terrain.length * terrain.horizontal_scale,
                         0.5 * terrain.width * terrain.horizontal_scale,
                         -0.5 * terrain.vertical_scale)
        ground_dim = (terrain.length * terrain.horizontal_scale,
                      terrain.width * terrain.horizontal_scale,
                      terrain.vertical_scale)
        ground_mesh = trimesh.creation.box(ground_dim, trimesh.transformations.translation_matrix(ground_center))
        meshes_list.append(ground_mesh)
        # concatenate all meshes to get the final terrain mesh
        terrain.terrain_mesh = trimesh.util.concatenate(meshes_list)
    
    return terrain

def high_platform_gaps_terrain(terrain : SubTerrain,
                    gap_size: float,
                    high_platform_height: float,
                    high_platform_length: float,
                    high_platform_width: float,
                    high_platform_distance_y: float,
                    platform_size: float = 1.,
                    depth: float = -10.,
                    terrain_type: str = None,
                    simplify_mesh: bool = False) -> SubTerrain:
    """Generate a terrain with a high platform and multiple gaps in the X direction track

    Args:
        terrain (SubTerrain): subterrain object to be updated
        gap_size (float): size of the gap (in meters)
        high_platform_height (float): height of the high platform (in meters, positive value)
        high_platform_length (float): length of the high platform (in meters)
        high_platform_width (float): width of the high platform (in meters)
        high_platform_distance_y (float): distance between high platforms in the y direction (in meters)
        platform_size (float, optional): size of the platform in the middle of the terrain. Defaults to 1..
        depth (float, optional): depth of the gap. Defaults to -10..
        terrain_type (str, optional): type of the terrain. Defaults to None.
        simplify_mesh (bool, optional): whether to simplify the terrain mesh. Defaults to False.
    Returns:
        SubTerrain: updated terrain with a gap and a high platform
    """
    if terrain_type in [None, "plane"]:
        raise ValueError("high_platform_gaps_terrain can only be used for heightfield or trimesh terrain type")
    
    gap_size = int(gap_size / terrain.horizontal_scale)
    high_platform_height = int(high_platform_height / terrain.vertical_scale)
    high_platform_length = int(high_platform_length / terrain.horizontal_scale)
    high_platform_width = int(high_platform_width / terrain.horizontal_scale)
    high_platform_distance_y = int(high_platform_distance_y / terrain.horizontal_scale)
    depth = int(depth / terrain.vertical_scale)
    platform_size = int(platform_size / terrain.horizontal_scale / 2)
    
    if terrain_type == "trimesh":
        # generate terrain mesh along with the heightfield simultaneously
        # initialize list of meshes
        meshes_list = list()
    
    # fill the terrain with the depth value first
    terrain.height_field_raw[:, :] = depth
    if terrain_type == "trimesh":
        # add bottom mesh for the gaps
        bottom_center = (terrain.length * terrain.horizontal_scale / 2, 
                         terrain.width * terrain.horizontal_scale / 2, 
                         depth * terrain.vertical_scale - 0.05)
        bottom_dim = (terrain.length * terrain.horizontal_scale, 
                     terrain.width * terrain.horizontal_scale,
                     0.1)
        bottom_mesh = trimesh.creation.box(bottom_dim, trimesh.transformations.translation_matrix(bottom_center))
        meshes_list.append(bottom_mesh)
    
    def generate_x_direction_track(terrain, y_start, y_end):
        # positive x direction track
        x_start = x_center + platform_size
        x_end = x_start + high_platform_length
        while x_start < terrain.length:
            terrain.height_field_raw[x_start:x_end, y_start:y_end] = high_platform_height
            if terrain_type == "trimesh":
                box_center = (0.5 * (x_start + x_end) * terrain.horizontal_scale,
                            0.5 * (y_start + y_end) * terrain.horizontal_scale,
                            0.5 * (high_platform_height + depth) * terrain.vertical_scale)
                box_dim = ((x_end - x_start) * terrain.horizontal_scale,
                        (y_end - y_start) * terrain.horizontal_scale,
                        (high_platform_height - depth) * terrain.vertical_scale)
                box_mesh = trimesh.creation.box(box_dim, trimesh.transformations.translation_matrix(box_center))
                meshes_list.append(box_mesh)
            x_start = x_end + gap_size
            x_end = min(x_start + high_platform_length, terrain.length)
            
        # negative x direction track
        x_end = x_center - platform_size
        x_start = x_end - high_platform_length
        while x_end > 0:
            terrain.height_field_raw[x_start:x_end, y_start:y_end] = high_platform_height
            if terrain_type == "trimesh":
                box_center = (0.5 * (x_start + x_end) * terrain.horizontal_scale,
                            0.5 * (y_start + y_end) * terrain.horizontal_scale,
                            0.5 * (high_platform_height + depth) * terrain.vertical_scale)
                box_dim = ((x_end - x_start) * terrain.horizontal_scale,
                        (y_end - y_start) * terrain.horizontal_scale,
                        (high_platform_height - depth) * terrain.vertical_scale)
                box_mesh = trimesh.creation.box(box_dim, trimesh.transformations.translation_matrix(box_center))
                meshes_list.append(box_mesh)
            x_end = x_start - gap_size
            x_start = max(x_end - high_platform_length, 0)
        
        return terrain
    
    
    # only generate high platforms on the x direction track
    x_center = terrain.length // 2
    y_center = terrain.width // 2
    # first generate the high platform and gaps in the middle of the x direction track
    y_start = y_center - high_platform_width // 2
    y_end = y_center + high_platform_width // 2
    terrain = generate_x_direction_track(terrain, y_start, y_end)
    
    # generate high platforms and gaps towards both sides of the y direction
    # positive y direction
    y_start = y_center + high_platform_width // 2 + high_platform_distance_y
    y_end = y_start + high_platform_width
    while y_start < y_center + platform_size:
        terrain = generate_x_direction_track(terrain, y_start, y_end)
        y_start = y_end + high_platform_distance_y
        y_end = y_start + high_platform_width
    # negative y direction
    y_end = y_center - high_platform_width // 2 - high_platform_distance_y
    y_start = y_end - high_platform_width
    while y_end > y_center - platform_size:
        terrain = generate_x_direction_track(terrain, y_start, y_end)
        y_end = y_start - high_platform_distance_y
        y_start = y_end - high_platform_width
    
    # ground platform
    x1 = terrain.length // 2 - platform_size
    x2 = terrain.length // 2 + platform_size
    y1 = terrain.width // 2 - platform_size
    y2 = terrain.width // 2 + platform_size
    terrain.height_field_raw[x1:x2, y1:y2] = 0
    
    # Identify the edge points of the terrain and set the edge mask accordingly
    edge_threshold = int(0.04 / terrain.vertical_scale) # 4cm height difference to identify edge points
    # if the height difference between a point and its 4-connected neighbors is greater than the threshold, then this point is an edge point
    for i in range(0, terrain.length - 1):
        for j in range(0, terrain.width - 1):
            if (abs(terrain.height_field_raw[i, j] - terrain.height_field_raw[i-1, j]) > edge_threshold or
                abs(terrain.height_field_raw[i, j] - terrain.height_field_raw[i+1, j]) > edge_threshold or
                abs(terrain.height_field_raw[i, j] - terrain.height_field_raw[i, j-1]) > edge_threshold or
                abs(terrain.height_field_raw[i, j] - terrain.height_field_raw[i, j+1]) > edge_threshold):
                terrain.edge_mask[i, j] = True
                
    if terrain_type == "trimesh":
      if not simplify_mesh:
        vertices, triangles = convert_heightfield_to_trimesh(terrain.height_field_raw, terrain.horizontal_scale, terrain.vertical_scale)
        terrain.terrain_mesh = trimesh.Trimesh(vertices=vertices, faces=triangles)
        # add a small offset to align the terrain mesh with the border
        translation = np.array([
                terrain.horizontal_scale / 2.0,
                terrain.horizontal_scale / 2.0,
                0
            ])
        terrain.terrain_mesh.apply_translation(translation)
      
      else:
        platform_center = (0.5 * (x1 + x2) * terrain.horizontal_scale,
                           0.5 * (y1 + y2) * terrain.horizontal_scale,
                           0.5 * depth * terrain.vertical_scale)
        platform_dim = (platform_size * 2 * terrain.horizontal_scale,
                        platform_size * 2 * terrain.horizontal_scale,
                        abs(depth) * terrain.vertical_scale)
        platform_mesh = trimesh.creation.box(platform_dim, trimesh.transformations.translation_matrix(platform_center))
        meshes_list.append(platform_mesh)
        # concatenate all meshes to get the final terrain mesh
        terrain.terrain_mesh = trimesh.util.concatenate(meshes_list)
    
    return terrain
    

#---------- Trimesh Terrain Functions ----------#
# The program will use terrains directly generated by below functions when the terrain type is set to "trimesh".
# The heightfield is only used for sampling the height of the terrain at any (x,y) location.

def make_border(
    size: Tuple[float, float], 
    inner_size: Tuple[float, float], 
    height: float, 
    position: Tuple[float, float, float]
) -> List[trimesh.Trimesh]:
    """Generate meshes for a rectangular border with a hole in the middle.

    .. code:: text

        +---------------------+
        |#####################|
        |##+---------------+##|
        |##|               |##|
        |##|               |##| length
        |##|               |##| (y-axis)
        |##|               |##|
        |##+---------------+##|
        |#####################|
        +---------------------+
              width (x-axis)

    Args:
        size: The length (along x) and width (along y) of the terrain (in m).
        inner_size: The inner length (along x) and width (along y) of the hole (in m).
        height: The height of the border (in m).
        position: The center of the border (in m).

    Returns:
        A list of trimesh.Trimesh objects that represent the border.
    """
    # compute thickness of the border
    thickness_x = (size[0] - inner_size[0]) / 2.0
    thickness_y = (size[1] - inner_size[1]) / 2.0
    # generate tri-meshes for the border
    # top/bottom border
    box_dims = (size[0], thickness_y, height)
    # -- top
    box_pos = (position[0], position[1] + inner_size[1] / 2.0 + thickness_y / 2.0, position[2])
    box_mesh_top = trimesh.creation.box(box_dims, trimesh.transformations.translation_matrix(box_pos))
    # -- bottom
    box_pos = (position[0], position[1] - inner_size[1] / 2.0 - thickness_y / 2.0, position[2])
    box_mesh_bottom = trimesh.creation.box(box_dims, trimesh.transformations.translation_matrix(box_pos))
    # left/right border
    box_dims = (thickness_x, inner_size[1], height)
    # -- left
    box_pos = (position[0] - inner_size[0] / 2.0 - thickness_x / 2.0, position[1], position[2])
    box_mesh_left = trimesh.creation.box(box_dims, trimesh.transformations.translation_matrix(box_pos))
    # -- right
    box_pos = (position[0] + inner_size[0] / 2.0 + thickness_x / 2.0, position[1], position[2])
    box_mesh_right = trimesh.creation.box(box_dims, trimesh.transformations.translation_matrix(box_pos))
    # return the tri-meshes
    return [box_mesh_left, box_mesh_right, box_mesh_top, box_mesh_bottom]


def mesh_pyramid_stairs_terrain(terrain : SubTerrain,
                                step_width : float,
                                step_height : float,
                                platform_size: float = 1.) -> trimesh.Trimesh:
    """Generate a terrain with a pyramid stair pattern.

    The terrain is a pyramid stair pattern which trims to a flat platform at the center of the terrain.


    Args:
        step_width (float): width of each step in the stairs (in m)
        step_height (float): height of each step in the stairs (in m)
        platform_size (float): size of the flat platform at the center of the terrain (in

    Returns:
        the tri-mesh of the terrain.
    """
    
    # compute number of steps in x and y direction
    num_steps_x = (terrain.length * terrain.horizontal_scale - platform_size) // (2 * step_width) + 1
    num_steps_y = (terrain.width * terrain.horizontal_scale - platform_size) // (2 * step_width) + 1
    # we take the minimum number of steps in x and y direction
    num_steps = int(min(num_steps_x, num_steps_y))

    # initialize list of meshes
    meshes_list = list()

    # generate the terrain
    # -- compute the position of the center of the terrain
    terrain_center = [0.5 * terrain.length * terrain.horizontal_scale, 
                      0.5 * terrain.width * terrain.horizontal_scale, 
                      0.0]
    terrain_size = (terrain.length * terrain.horizontal_scale, 
                    terrain.width * terrain.horizontal_scale)
    # -- generate the stair pattern
    for k in range(num_steps):
        box_size = (terrain_size[0] - 2 * k * step_width, 
                    terrain_size[1] - 2 * k * step_width)
        # compute the quantities of the box
        # -- location
        box_z = terrain_center[2] + k * step_height / 2.0
        box_offset = (k + 0.5) * step_width
        # -- dimensions
        box_height = k * step_height
        # generate the boxes
        # top/bottom
        box_dims = (box_size[0], step_width, box_height)
        # -- top
        box_pos = (terrain_center[0], terrain_center[1] + terrain_size[1] / 2.0 - box_offset, box_z)
        box_top = trimesh.creation.box(box_dims, trimesh.transformations.translation_matrix(box_pos))
        # -- bottom
        box_pos = (terrain_center[0], terrain_center[1] - terrain_size[1] / 2.0 + box_offset, box_z)
        box_bottom = trimesh.creation.box(box_dims, trimesh.transformations.translation_matrix(box_pos))
        # right/left
        box_dims = (step_width, box_size[1] - 2 * step_width, box_height)
        # -- right
        box_pos = (terrain_center[0] + terrain_size[0] / 2.0 - box_offset, terrain_center[1], box_z)
        box_right = trimesh.creation.box(box_dims, trimesh.transformations.translation_matrix(box_pos))
        # -- left
        box_pos = (terrain_center[0] - terrain_size[0] / 2.0 + box_offset, terrain_center[1], box_z)
        box_left = trimesh.creation.box(box_dims, trimesh.transformations.translation_matrix(box_pos))
        # add the boxes to the list of meshes
        meshes_list += [box_top, box_bottom, box_right, box_left]

    # generate final box for the middle of the terrain
    box_dims = (
        terrain_size[0] - 2 * num_steps * step_width,
        terrain_size[1] - 2 * num_steps * step_width,
        (num_steps - 1) * step_height,
    )
    box_pos = (terrain_center[0], terrain_center[1], 
               terrain_center[2] + (num_steps - 1) * step_height / 2)
    box_middle = trimesh.creation.box(box_dims, trimesh.transformations.translation_matrix(box_pos))
    meshes_list.append(box_middle)
    mesh = trimesh.util.concatenate(meshes_list)
    
    return mesh

def mesh_inverted_pyramid_stairs_terrain(terrain : SubTerrain,
                                        step_width : float,
                                        step_height : float,
                                        platform_size: float = 1.) -> trimesh.Trimesh:
    """Generate a terrain with a inverted pyramid stair pattern.

    The terrain is an inverted pyramid stair pattern which trims to a flat platform at the center of the terrain.

    Args:
        terrain: The terrain configuration.
        step_width: The width of each step.
        step_height: The height of each step.
        platform_size: The size of the flat platform at the center of the terrain.

    Returns:
        The tri-mesh of the terrain.
    """
    # compute number of steps in x and y direction
    num_steps_x = (terrain.length * terrain.horizontal_scale - platform_size) // (2 * step_width) + 1
    num_steps_y = (terrain.width * terrain.horizontal_scale - platform_size) // (2 * step_width) + 1
    # we take the minimum number of steps in x and y direction
    num_steps = int(min(num_steps_x, num_steps_y))
    # total height of the terrain
    total_height = (num_steps - 1) * step_height

    # initialize list of meshes
    meshes_list = list()

    # generate the terrain
    # -- compute the position of the center of the terrain
    terrain_center = [0.5 * terrain.length * terrain.horizontal_scale, 
                      0.5 * terrain.width * terrain.horizontal_scale, 
                      0.0]
    terrain_size = (terrain.length * terrain.horizontal_scale, 
                    terrain.width * terrain.horizontal_scale)
    # -- generate the stair pattern
    for k in range(num_steps):
        box_size = (terrain_size[0] - 2 * k * step_width, 
                    terrain_size[1] - 2 * k * step_width)
        # compute the quantities of the box
        # -- location
        box_z = terrain_center[2] - total_height / 2 - k * step_height / 2.0
        box_offset = (k + 0.5) * step_width
        # -- dimensions
        box_height = total_height - k * step_height
        # generate the boxes
        # top/bottom
        box_dims = (box_size[0], step_width, box_height)
        # -- top
        box_pos = (terrain_center[0], terrain_center[1] + terrain_size[1] / 2.0 - box_offset, box_z)
        box_top = trimesh.creation.box(box_dims, trimesh.transformations.translation_matrix(box_pos))
        # -- bottom
        box_pos = (terrain_center[0], terrain_center[1] - terrain_size[1] / 2.0 + box_offset, box_z)
        box_bottom = trimesh.creation.box(box_dims, trimesh.transformations.translation_matrix(box_pos))
        # right/left
        box_dims = (step_width, box_size[1] - 2 * step_width, box_height)
        # -- right
        box_pos = (terrain_center[0] + terrain_size[0] / 2.0 - box_offset, terrain_center[1], box_z)
        box_right = trimesh.creation.box(box_dims, trimesh.transformations.translation_matrix(box_pos))
        # -- left
        box_pos = (terrain_center[0] - terrain_size[0] / 2.0 + box_offset, terrain_center[1], box_z)
        box_left = trimesh.creation.box(box_dims, trimesh.transformations.translation_matrix(box_pos))
        # add the boxes to the list of meshes
        meshes_list += [box_top, box_bottom, box_right, box_left]
    # generate final box for the middle of the terrain
    box_dims = (
        terrain_size[0] - 2 * num_steps * step_width,
        terrain_size[1] - 2 * num_steps * step_width,
        step_height,
    )
    box_pos = (terrain_center[0], 
               terrain_center[1], 
               terrain_center[2] - total_height - step_height / 2)
    box_middle = trimesh.creation.box(box_dims, trimesh.transformations.translation_matrix(box_pos))
    meshes_list.append(box_middle)
    mesh = trimesh.util.concatenate(meshes_list)

    return mesh

def mesh_gap_terrain(terrain : SubTerrain,
                     gap_size : float,
                     platform_size : float = 1.) -> trimesh.Trimesh:
    """Generate a terrain with a gap around the platform.

    The terrain has a ground with a platform in the middle. The platform is surrounded by a gap
    of width :obj:`gap_width` on all sides.

    Args:
        terrain (SubTerrain): The terrain configuration.
        gap_size (float): The width of the gap around the platform (in m).
        platform_size (float): The size of the flat platform at the center of the terrain (in m).

    Returns:
        Trimesh: The tri-mesh of the terrain.
    """
    # initialize list of meshes
    meshes_list = list()
    # constants for terrain generation
    terrain_height = 1.0
    terrain_center = (0.5 * terrain.length * terrain.horizontal_scale, 
                      0.5 * terrain.width * terrain.horizontal_scale, 
                      -terrain_height / 2)
    terrain_size = (terrain.length * terrain.horizontal_scale, 
                    terrain.width * terrain.horizontal_scale)

    # Generate the outer ring
    inner_size = (platform_size + 2 * gap_size, 
                  platform_size + 2 * gap_size)
    meshes_list += make_border(terrain_size, 
                               inner_size, 
                               terrain_height, 
                               terrain_center)
    # Generate the inner box
    box_dim = (platform_size, platform_size, terrain_height)
    box = trimesh.creation.box(box_dim, trimesh.transformations.translation_matrix(terrain_center))
    meshes_list.append(box)
    mesh = trimesh.util.concatenate(meshes_list)

    return mesh

def mesh_pit_terrain(terrain : SubTerrain, 
                     depth : float, 
                     platform_size: float = 1.) -> trimesh.Trimesh:
    """Generate a terrain with a pit with levels (stairs) leading out of the pit.

    The terrain contains a platform at the center and a staircase leading out of the pit.
    The staircase is a series of steps that are aligned along the x- and y- axis. The steps are
    created by extruding a ring along the x- and y- axis. If :obj:`is_double_pit` is True, the pit
    contains two levels.

    Args:
        terrain (SubTerrain): The terrain configuration.
        depth (float): The depth of the pit (in m).
        platform_size (float): The size of the flat platform at the center of the terrain (in m).

    Returns:
        Trimesh: The tri-mesh of the terrain.
    """
    # resolve the terrain configuration
    pit_depth = depth

    # initialize list of meshes
    meshes_list = list()
    # extract quantities
    inner_pit_size = (platform_size, 
                      platform_size)
    total_depth = pit_depth
    # constants for terrain generation
    terrain_height = 1.0

    # generate the pit (outer ring)
    pit_center = [0.5 * terrain.length * terrain.horizontal_scale, 
                  0.5 * terrain.width * terrain.horizontal_scale, 
                  -total_depth * 0.5]
    terrain_size = (terrain.length * terrain.horizontal_scale, 
                    terrain.width * terrain.horizontal_scale)
    meshes_list += make_border(terrain_size, 
                               inner_pit_size, 
                               total_depth, 
                               pit_center)
    # generate the ground
    dim = (terrain.length * terrain.horizontal_scale, 
           terrain.width * terrain.horizontal_scale, 
           terrain_height)
    pos = (0.5 * terrain.length * terrain.horizontal_scale, 
           0.5 * terrain.width * terrain.horizontal_scale, 
           -total_depth - terrain_height / 2)
    ground_meshes = trimesh.creation.box(dim, trimesh.transformations.translation_matrix(pos))
    meshes_list.append(ground_meshes)
    mesh = trimesh.util.concatenate(meshes_list)

    return mesh


#---------- Utility Functions ----------#

def convert_heightfield_to_trimesh(height_field_raw : np.ndarray, 
                                   horizontal_scale: float, 
                                   vertical_scale: float,
                                   slope_threshold = None):
    """
    Convert a heightfield array to a triangle mesh represented by vertices and triangles.
    Optionally, corrects vertical surfaces above the provide slope threshold:

        If (y2-y1)/(x2-x1) > slope_threshold -> Move A to A' (set x1 = x2). Do this for all directions.
                   B(x2,y2)
                  /|
                 / |
                /  |
        (x1,y1)A---A'(x2',y1)

    Parameters:
        height_field_raw (np.array): input heightfield
        horizontal_scale (float): horizontal scale of the heightfield [meters]
        vertical_scale (float): vertical scale of the heightfield [meters]
        slope_threshold (float): the slope threshold above which surfaces are made vertical. If None no correction is applied (default: None)
    Returns:
        vertices (np.array(float)): array of shape (num_vertices, 3). Each row represents the location of each vertex [meters]
        triangles (np.array(int)): array of shape (num_triangles, 3). Each row represents the indices of the 3 vertices connected by this triangle.
    """
    hf = height_field_raw
    num_rows = hf.shape[0]
    num_cols = hf.shape[1]

    y = np.linspace(0, (num_cols-1)*horizontal_scale, num_cols)
    x = np.linspace(0, (num_rows-1)*horizontal_scale, num_rows)
    yy, xx = np.meshgrid(y, x)

    if slope_threshold is not None:

        slope_threshold *= horizontal_scale / vertical_scale
        move_x = np.zeros((num_rows, num_cols))
        move_y = np.zeros((num_rows, num_cols))
        move_corners = np.zeros((num_rows, num_cols))
        move_x[:num_rows-1, :] += (hf[1:num_rows, :] - hf[:num_rows-1, :] > slope_threshold)
        move_x[1:num_rows, :] -= (hf[:num_rows-1, :] - hf[1:num_rows, :] > slope_threshold)
        move_y[:, :num_cols-1] += (hf[:, 1:num_cols] - hf[:, :num_cols-1] > slope_threshold)
        move_y[:, 1:num_cols] -= (hf[:, :num_cols-1] - hf[:, 1:num_cols] > slope_threshold)
        move_corners[:num_rows-1, :num_cols-1] += (hf[1:num_rows, 1:num_cols] - hf[:num_rows-1, :num_cols-1] > slope_threshold)
        move_corners[1:num_rows, 1:num_cols] -= (hf[:num_rows-1, :num_cols-1] - hf[1:num_rows, 1:num_cols] > slope_threshold)
        xx += (move_x + move_corners*(move_x == 0)) * horizontal_scale
        yy += (move_y + move_corners*(move_y == 0)) * horizontal_scale

    # create triangle mesh vertices and triangles from the heightfield grid
    vertices = np.zeros((num_rows*num_cols, 3), dtype=np.float32)
    vertices[:, 0] = xx.flatten()
    vertices[:, 1] = yy.flatten()
    vertices[:, 2] = hf.flatten() * vertical_scale
    triangles = -np.ones((2*(num_rows-1)*(num_cols-1), 3), dtype=np.uint32)
    for i in range(num_rows - 1):
        ind0 = np.arange(0, num_cols-1) + i*num_cols
        ind1 = ind0 + 1
        ind2 = ind0 + num_cols
        ind3 = ind2 + 1
        start = 2*i*(num_cols-1)
        stop = start + 2*(num_cols-1)
        triangles[start:stop:2, 0] = ind0
        triangles[start:stop:2, 1] = ind3
        triangles[start:stop:2, 2] = ind1
        triangles[start+1:stop:2, 0] = ind0
        triangles[start+1:stop:2, 1] = ind2
        triangles[start+1:stop:2, 2] = ind3

    return vertices, triangles