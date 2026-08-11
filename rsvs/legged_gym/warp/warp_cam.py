# import nvtx
import warp as wp
import math
import torch

from .warp_kernels.warp_camera_kernels import (
    DepthCameraWarpKernels,
)


class WarpCam:
    def __init__(self, tensor_dict, num_envs, config, mesh_ids_array, device="cuda:0"):
        self.cfg = config
        self.tensor_dict = tensor_dict
        self.num_envs = num_envs
        self.num_sensors = self.cfg.depth_camera_config.num_sensors
        self.mesh_ids_array = mesh_ids_array

        self.width = self.cfg.depth_camera_config.resolution[1]
        self.height = self.cfg.depth_camera_config.resolution[0]

        self.horizontal_fov = math.radians(self.cfg.depth_camera_config.horizontal_fov_deg)
        self.far_plane = self.cfg.depth_camera_config.far_plane
        self.calculate_depth = self.cfg.depth_camera_config.calculate_depth
        self.device = device
        
        self.depth_image_tensor = tensor_dict["depth_image_tensor"]
        self.camera_pos_tensor = tensor_dict["sensor_pos_tensor"]
        self.camera_orientation_tensor = tensor_dict["sensor_quat_tensor"]

        self.graph = None

        self.initialize_camera_matrices()
        self.init_tensors()

    def initialize_camera_matrices(self):
        # Calculate camera params
        W = self.width
        H = self.height
        # principal point at the center of the image
        (c_x, c_y) = (W / 2, H / 2)
        # focal length in pixels, implicitly assuming dx(pixel length)=1
        f = W / 2 * 1 / math.tan(self.horizontal_fov / 2)

        vertical_fov = 2 * math.atan(H / (2 * f))
        # f_x equals to f because f is already in pixel unit
        f_x = c_x / math.tan(self.horizontal_fov / 2)
        # f_y equals to f, assuming square pixels
        f_y = c_y / math.tan(vertical_fov / 2)

        # simple pinhole model
        # transformation matrix from camera frame to pixel frame, emitting ray toward +z direction
        # self.K = wp.mat44( # row first
        #     f_x, 0.0, c_x, 0.0,
        #     0.0, f_y, c_y, 0.0,
        #     0.0, 0.0, 1.0, 0.0,
        #     0.0, 0.0, 0.0, 1.0,
        # )
        # rectify pixel frame
        self.K = wp.mat44( # row first
            0.0, -f_x, c_x, 0.0,
            f_y, 0.0, c_y, 0.0,
            0.0, 0.0, 1.0, 0.0,
            0.0, 0.0, 0.0, 1.0,
        )
        self.K_inv = wp.inverse(self.K)

        self.c_x = int(c_x)
        self.c_y = int(c_y)

    def create_render_graph_pointcloud(self, debug=False):
        if not debug:
            print(f"creating render graph")
            wp.capture_begin(device=self.device)
        # with wp.ScopedTimer("render"):
        if self.cfg.depth_camera_config.segmentation_camera == True:
            wp.launch(
                kernel=DepthCameraWarpKernels.draw_optimized_kernel_pointcloud_segmentation,
                dim=(self.num_envs, self.num_sensors, self.width, self.height),
                inputs=[
                    self.mesh_ids_array,
                    self.camera_position_array,
                    self.camera_orientation_array,
                    self.K_inv,
                    self.far_plane,
                    self.pixels,
                    self.segmentation_pixels,
                    self.c_x,
                    self.c_y,
                    self.pointcloud_in_world_frame,
                ],
                device=self.device,
            )
        else:
            wp.launch(
                kernel=DepthCameraWarpKernels.draw_optimized_kernel_pointcloud,
                dim=(self.num_envs, self.num_sensors, self.width, self.height),
                inputs=[
                    self.mesh_ids_array,
                    self.camera_position_array,
                    self.camera_orientation_array,
                    self.K_inv,
                    self.far_plane,
                    self.pixels,
                    self.c_x,
                    self.c_y,
                    self.pointcloud_in_world_frame,
                ],
                device=self.device,
            )
        if not debug:
            print(f"finishing capture of render graph")
            self.graph = wp.capture_end(device=self.device)

    def create_render_graph_depth_range(self, debug=False):
        if not debug:
            print(f"creating render graph")
            wp.capture_begin(device=self.device)
        # with wp.ScopedTimer("render"):
        if self.cfg.depth_camera_config.segmentation_camera == True:
            wp.launch(
                kernel=DepthCameraWarpKernels.draw_optimized_kernel_depth_range_segmentation,
                dim=(self.num_envs, self.num_sensors, self.width, self.height),
                inputs=[
                    self.mesh_ids_array,
                    self.camera_position_array,
                    self.camera_orientation_array,
                    self.K_inv,
                    self.far_plane,
                    self.pixels,
                    self.segmentation_pixels,
                    self.c_x,
                    self.c_y,
                    self.calculate_depth,
                ],
                device=self.device,
            )
        else:
            wp.launch(
                kernel=DepthCameraWarpKernels.draw_optimized_kernel_depth_range,
                dim=(self.num_envs, self.num_sensors, self.width, self.height),
                inputs=[
                    self.mesh_ids_array,
                    self.camera_position_array,
                    self.camera_orientation_array,
                    self.K_inv,
                    self.far_plane,
                    self.pixels,
                    self.c_x,
                    self.c_y,
                    self.calculate_depth,
                ],
                device=self.device,
            )
        if not debug:
            print(f"finishing capture of render graph")
            self.graph = wp.capture_end(device=self.device)

    def init_tensors(self):
        # init buffers. None when uninitialized
        if self.cfg.depth_camera_config.return_pointcloud:
            self.pixels = wp.from_torch(self.depth_image_tensor, dtype=wp.vec3)
            self.pointcloud_in_world_frame = self.cfg.depth_camera_config.pointcloud_in_world_frame
        else:
            self.pixels = wp.from_torch(self.depth_image_tensor, dtype=wp.float32)

        # if self.cfg.depth_camera_config.segmentation_camera == True:
        #     self.segmentation_pixels = wp.from_torch(segmentation_pixels, dtype=wp.int32)
        # else:
        #     self.segmentation_pixels = segmentation_pixels
        self.segmentation_pixels = None
        
        self.camera_position_array = wp.from_torch(self.camera_pos_tensor.view(self.num_envs, 1, 3), dtype=wp.vec3)
        self.camera_orientation_array = wp.from_torch(
            self.camera_orientation_tensor.view(self.num_envs, 1, 4), dtype=wp.quat)
    
    def tensor_indices_to_slice(idx: torch.Tensor):
        # expects 1-D int tensor
        idx = idx.to(dtype=torch.long)
        if idx.numel() == 0:
            return slice(0, 0)
        sorted_idx, _ = torch.sort(idx)
        first = int(sorted_idx[0].item())
        last = int(sorted_idx[-1].item())
        if last - first + 1 == sorted_idx.numel():
            return slice(first, last + 1)  # stop is exclusive
        return None  # not contiguous
    
    def reset(self, env_ids=None, value=0.0):
        if env_ids == None:
            self.pixels.fill_(value)
            return
        
        if isinstance(env_ids, int):
            self.pixels[env_ids].fill_(value)
            return
        
        if isinstance(env_ids, slice):
            self.pixels[env_ids].fill_(value)
            return
        
        if isinstance(env_ids, torch.Tensor):
            if env_ids.numel() == 0:
                return
            sl = self.tensor_indices_to_slice(env_ids)
            if sl is not None:
                self.pixels[sl].fill_(value)
            else:
                for i in env_ids.tolist():
                    self.pixels[int(i)].fill_(value)
            return

        raise TypeError(f"Unsupported type for env_ids: {type(env_ids)}")
        
    # @nvtx.annotate()
    def update(self, debug=False):
        if self.graph is None:
            if self.cfg.depth_camera_config.return_pointcloud:
                self.create_render_graph_pointcloud(debug=debug)
            else:
                self.create_render_graph_depth_range(debug=debug)

        if self.graph is not None:
            wp.capture_launch(self.graph)

        return wp.to_torch(self.pixels)
