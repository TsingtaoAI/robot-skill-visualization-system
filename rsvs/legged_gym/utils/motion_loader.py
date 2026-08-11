import glob
import sys
from legged_gym import LEGGED_GYM_ROOT_DIR
from legged_gym.utils.math_utils import quat_slerp, standardize_quaternion, quat_rotate_inverse_np

import torch
import numpy as np

import pickle

# Compatibility shim: pickle files saved with NumPy >= 2.0 reference
# numpy._core (e.g. numpy._core.multiarray), which does not exist in
# NumPy < 2.0.  Register aliases so unpickling works transparently.
if not hasattr(np, '_core'):
    import numpy.core as _np_core
    for _attr in dir(_np_core):
        _mod_name = f"numpy._core.{_attr}"
        _submod = getattr(_np_core, _attr, None)
        if isinstance(_submod, type(sys)):
            sys.modules.setdefault(_mod_name, _submod)
    sys.modules.setdefault("numpy._core", _np_core)
    sys.modules.setdefault("numpy._core.multiarray", _np_core.multiarray)
    del _np_core, _attr, _mod_name, _submod

class AMPLoader:

    def __init__(
            self,
            device,
            num_dof,
            num_key_bodies,
            time_between_frames,
            preload_transitions=False,
            num_preload_transitions=1000000,
            motion_files=glob.glob(f'{LEGGED_GYM_ROOT_DIR}/resources/reference_motion/*'),
            ):
        """Expert dataset provides AMP observations from mocap dataset.
        Args:
            device: torch device to store the loaded trajectories and preloaded transitions.
            num_dof: number of degrees of freedom of the robot, used to parse the motion data.
            num_key_bodies: number of key body points to be included in the AMP observations, used to parse the motion data.
            time_between_frames: Amount of time in seconds between transition.
            preload_transitions: Whether to preload a set of transitions from the trajectories for faster sampling during training. If False, transitions will be sampled on the fly.
            num_preload_transitions: If preloading transitions, how many transitions to preload.
            motion_files: List of file paths to the motion data files. Each file should be a .pkl file containing a dictionary with keys "root_pos", "root_rot", "root_lin_vel", "root_ang_vel", "dof_pos", "dof_vel", and "key_body_pos_relative_to_base", where the values are numpy arrays of shape (num_frames, feature_dim) corresponding to the trajectory of that feature over time.
        
        """
        self.device = device
        self.time_between_frames = time_between_frames
        # size of different parts of the observation
        self.base_pos_size = 3
        self.base_rot_size = 4
        self.base_lin_vel_size = 3
        self.base_ang_vel_size = 3
        self.dof_pos_size = num_dof
        self.dof_vel_size = num_dof
        self.key_body_pos_size = num_key_bodies * 3
        # indices for slicing the observation tensor
        self.base_pos_start_idx = 0
        self.base_pos_end_idx = self.base_pos_start_idx + self.base_pos_size
        self.base_rot_start_idx = self.base_pos_end_idx
        self.base_rot_end_idx = self.base_rot_start_idx + self.base_rot_size
        self.base_lin_vel_start_idx = self.base_rot_end_idx
        self.base_lin_vel_end_idx = self.base_lin_vel_start_idx + self.base_lin_vel_size
        self.base_ang_vel_start_idx = self.base_lin_vel_end_idx
        self.base_ang_vel_end_idx = self.base_ang_vel_start_idx + self.base_ang_vel_size
        self.dof_pos_start_idx = self.base_ang_vel_end_idx
        self.dof_pos_end_idx = self.dof_pos_start_idx + self.dof_pos_size
        self.dof_vel_start_idx = self.dof_pos_end_idx
        self.dof_vel_end_idx = self.dof_vel_start_idx + self.dof_vel_size
        self.key_body_pos_start_idx = self.dof_vel_end_idx
        self.key_body_pos_end_idx = self.key_body_pos_start_idx + self.key_body_pos_size
        
        self.global_gravity = np.array([0.0, 0.0, -1.0], dtype=np.float32)
        
        # Values to store for each trajectory.
        self.trajectories = []
        self.trajectories_full = []
        self.trajectory_names = []
        self.trajectory_idxs = []
        self.trajectory_lens = []  # Traj length in seconds.
        self.trajectory_weights = []
        self.trajectory_frame_durations = [] # time between frames of reference motion
        self.trajectory_num_frames = []

        # open the .pkl file and load the motion data
        # the motion data should be a dictionary with the following keys:
        # motion_data = {
        #     "fps": aligned_fps,
        #     "root_pos": root_pos.cpu().numpy(), # root link position in world frame
        #     "root_lin_vel": root_lin_vel.cpu().numpy(), # root link linear velocity in world frame
        #     "root_rot": root_rot.cpu().numpy(), # root link orientation as quaternion in world frame, xyzw order
        #     "root_euler": root_euler.cpu().numpy(), # root link orientation as euler angles in world frame
        #     "root_ang_vel": root_ang_vel.cpu().numpy(), # root link angular velocity in world frame
        #     "dof_pos": dof_pos.cpu().numpy(), # joint angles matching dof_names order
        #     "dof_vel": dof_vel.cpu().numpy(), # joint velocities matching dof_names order
        #     "key_body_pos_relative_to_base": key_body_pos_relative_to_base.cpu().numpy(),
        #           # key body point positions relative to the base in the world frame, shape [motion_length, num_key_bodies, 3]
        # }
        # Load all motions 
        for i, motion_file in enumerate(motion_files):
            self.trajectory_names.append(motion_file.split('.')[0])
            with open(motion_file, "rb") as f:
                motion_data_all = pickle.load(f)
            root_pos_data = motion_data_all["root_pos"]
            root_rot_data = motion_data_all["root_rot"]
            root_lin_vel_data = motion_data_all["root_lin_vel"]
            root_lin_vel_data = quat_rotate_inverse_np(root_rot_data, root_lin_vel_data)
            root_ang_vel_data = motion_data_all["root_ang_vel"]
            root_ang_vel_data = quat_rotate_inverse_np(root_rot_data, root_ang_vel_data)
            dof_pos_data = motion_data_all["dof_pos"]
            dof_vel_data = motion_data_all["dof_vel"]
            key_body_pos_relative_to_base_data = motion_data_all["key_body_pos_relative_to_base"]
            motion_data = np.concatenate([
                root_pos_data,
                root_rot_data,
                root_lin_vel_data,
                root_ang_vel_data,
                # exclude head joint and head link
                dof_pos_data, 
                dof_vel_data,
                key_body_pos_relative_to_base_data.reshape(key_body_pos_relative_to_base_data.shape[0], -1)
            ], axis=-1)
                
            # store the trajectory and its metadata
            # Only use base_lin_vel, base_ang_vel, dof_pos, dof_vel and key_body_pos_relative_to_base for AMP learning
            self.trajectories.append(torch.tensor(
                motion_data[:, self.base_lin_vel_start_idx:], dtype=torch.float32, device=device))
            # Full trajectory is used for state initialization.
            self.trajectories_full.append(torch.tensor(
                motion_data, dtype=torch.float32, device=device))
            self.trajectory_idxs.append(i)
            self.trajectory_weights.append(
                    1.0) # ! Use uniform weights for now, but can be changed to prioritize some trajectories over others.
            frame_duration = float(1.0 / motion_data_all["fps"])
            self.trajectory_frame_durations.append(frame_duration)
            traj_len = (motion_data.shape[0] - 1) * frame_duration
            self.trajectory_lens.append(traj_len)
            self.trajectory_num_frames.append(float(motion_data.shape[0]))

            print(f"Loaded {traj_len}s. motion from {motion_file}.")
        
        # Trajectory weights are used to sample some trajectories more than others.
        self.trajectory_weights = np.array(self.trajectory_weights) / np.sum(self.trajectory_weights)
        self.trajectory_frame_durations = np.array(self.trajectory_frame_durations)
        self.trajectory_lens = np.array(self.trajectory_lens)
        self.trajectory_num_frames = np.array(self.trajectory_num_frames)

        # Preload transitions.
        self.preload_transitions = preload_transitions
        if self.preload_transitions:
            print(f'Preloading {num_preload_transitions} transitions')
            traj_idxs = self.weighted_traj_idx_sample_batch(num_preload_transitions)
            times = self.traj_time_sample_batch(traj_idxs)
            self.preloaded_s = self.get_full_frame_at_time_batch(traj_idxs, times)
            self.preloaded_s_next = self.get_full_frame_at_time_batch(traj_idxs, times + self.time_between_frames)
            print(f'Finished preloading')

    def weighted_traj_idx_sample(self):
        """Get traj idx via weighted sampling."""
        return np.random.choice(
            self.trajectory_idxs, p=self.trajectory_weights)

    def weighted_traj_idx_sample_batch(self, size):
        """Batch sample traj idxs."""
        return np.random.choice(
            self.trajectory_idxs, size=size, p=self.trajectory_weights,
            replace=True)

    def traj_time_sample(self, traj_idx):
        """Sample random time for traj.
            traj_idx is a single integer
        """
        subst = self.time_between_frames + self.trajectory_frame_durations[traj_idx]
        return max(
            0, (self.trajectory_lens[traj_idx] * np.random.uniform() - subst))

    def traj_time_sample_batch(self, traj_idxs):
        """Sample random time for multiple trajectories."""
        # subst = time_between_frames + trajectory_frame_durations[traj_idx] is the minimum “margin” you need at the end of the motion: 
        # time_between_frames because later you will query at t and t + time_between_frames (for s and s_next), 
        # and trajectory_frame_durations because frame indexing / interpolation near the last frame needs at least one frame interval of slack.
        # Without subtracting this offset, u * trajectory_lens could be arbitrarily close to the total trajectory length. 
        # Then t + time_between_frames (and the interpolation between neighboring frames) could require accessing beyond the last valid frame index, 
        # causing out-of-bounds or degenerate interpolation at the very end.
        subst = self.time_between_frames + self.trajectory_frame_durations[traj_idxs]
        time_samples = self.trajectory_lens[traj_idxs] * np.random.uniform(size=len(traj_idxs)) - subst
        return np.maximum(np.zeros_like(time_samples), time_samples)

    def slerp(self, val0, val1, blend):
        # Linearly interpolate between two values.
        return (1.0 - blend) * val0 + blend * val1

    def get_frame_at_time(self, traj_idx, time):
        """Returns frame for the given trajectory at the specified time."""
        # Note that the frame returned is not necessarily an actual frame from the dataset, but may be an interpolation of two frames. This allows for more diverse sampling of states from the reference motions.
        # The interpolation is done by finding the two frames in the trajectory that the specified time falls between, and then blending those two frames together based on how close the specified time is to each frame. The blending is a simple linear interpolation (slerp) of the two frames.
        # normalize time to [0, 1] based on trajectory length
        p = float(time) / self.trajectory_lens[traj_idx]
        # length of trajectory in frames
        n = self.trajectories[traj_idx].shape[0]
        # find the two frames that the specified time falls between
        idx_low, idx_high = int(np.floor(p * n)), int(np.ceil(p * n))
        frame_start = self.trajectories[traj_idx][idx_low]
        frame_end = self.trajectories[traj_idx][idx_high]
        # blend based on how close the specified time is to each frame
        blend = p * n - idx_low
        return self.slerp(frame_start, frame_end, blend)

    def get_frame_at_time_batch(self, traj_idxs, times):
        """Returns frame for the given trajectory at the specified time."""
        p = times / self.trajectory_lens[traj_idxs]
        n = self.trajectory_num_frames[traj_idxs]
        idx_low, idx_high = np.floor(p * n).astype(np.int64), np.ceil(p * n).astype(np.int64)
        all_frame_starts = torch.zeros(len(traj_idxs), self.observation_dim, device=self.device)
        all_frame_ends = torch.zeros(len(traj_idxs), self.observation_dim, device=self.device)
        for traj_idx in set(traj_idxs):
            trajectory = self.trajectories[traj_idx]
            traj_mask = traj_idxs == traj_idx
            all_frame_starts[traj_mask] = trajectory[idx_low[traj_mask]]
            all_frame_ends[traj_mask] = trajectory[idx_high[traj_mask]]
        blend = torch.tensor(p * n - idx_low, device=self.device, dtype=torch.float32).unsqueeze(-1)
        return self.slerp(all_frame_starts, all_frame_ends, blend)

    def get_full_frame_at_time(self, traj_idx, time):
        """Returns full frame for the given trajectory at the specified time."""
        p = float(time) / self.trajectory_lens[traj_idx]
        n = self.trajectories_full[traj_idx].shape[0]
        idx_low, idx_high = int(np.floor(p * n)), int(np.ceil(p * n))
        frame_start = self.trajectories_full[traj_idx][idx_low]
        frame_end = self.trajectories_full[traj_idx][idx_high]
        blend = p * n - idx_low
        return self.blend_frame_pose(frame_start, frame_end, blend)

    def get_full_frame_at_time_batch(self, traj_idxs, times):
        p = times / self.trajectory_lens[traj_idxs]
        n = self.trajectory_num_frames[traj_idxs]
        idx_low, idx_high = np.floor(p * n).astype(np.int64), np.ceil(p * n).astype(np.int64)
        all_frame_pos_starts = torch.zeros(len(traj_idxs), self.base_pos_size, device=self.device)
        all_frame_pos_ends = torch.zeros(len(traj_idxs), self.base_pos_size, device=self.device)
        all_frame_rot_starts = torch.zeros(len(traj_idxs), self.base_rot_size, device=self.device)
        all_frame_rot_ends = torch.zeros(len(traj_idxs), self.base_rot_size, device=self.device)
        all_frame_amp_starts = torch.zeros(len(traj_idxs), self.key_body_pos_end_idx - self.base_rot_end_idx, device=self.device)
        all_frame_amp_ends = torch.zeros(len(traj_idxs), self.key_body_pos_end_idx - self.base_rot_end_idx, device=self.device)
        for traj_idx in set(traj_idxs):
            trajectory = self.trajectories_full[traj_idx]
            traj_mask = traj_idxs == traj_idx
            all_frame_pos_starts[traj_mask] = self.get_base_pos_batch(trajectory[idx_low[traj_mask]])
            all_frame_pos_ends[traj_mask] = self.get_base_pos_batch(trajectory[idx_high[traj_mask]])
            all_frame_rot_starts[traj_mask] = self.get_base_rot_batch(trajectory[idx_low[traj_mask]])
            all_frame_rot_ends[traj_mask] = self.get_base_rot_batch(trajectory[idx_high[traj_mask]])
            all_frame_amp_starts[traj_mask] = trajectory[idx_low[traj_mask]][:, self.base_lin_vel_start_idx:self.key_body_pos_end_idx]
            all_frame_amp_ends[traj_mask] = trajectory[idx_high[traj_mask]][:, self.base_lin_vel_start_idx:self.key_body_pos_end_idx]
        blend = torch.tensor(p * n - idx_low, device=self.device, dtype=torch.float32).unsqueeze(-1)

        pos_blend = self.slerp(all_frame_pos_starts, all_frame_pos_ends, blend)
        rot_blend = quat_slerp(all_frame_rot_starts, all_frame_rot_ends, blend)
        rot_blend = standardize_quaternion(rot_blend)
        amp_blend = self.slerp(all_frame_amp_starts, all_frame_amp_ends, blend)
        return torch.cat([pos_blend, rot_blend, amp_blend], dim=-1)

    def get_frame(self):
        """Returns random frame."""
        traj_idx = self.weighted_traj_idx_sample()
        sampled_time = self.traj_time_sample(traj_idx)
        return self.get_frame_at_time(traj_idx, sampled_time)

    def get_full_frame(self):
        """Returns random full frame."""
        traj_idx = self.weighted_traj_idx_sample()
        sampled_time = self.traj_time_sample(traj_idx)
        return self.get_full_frame_at_time(traj_idx, sampled_time)

    def get_full_frame_batch(self, num_frames):
        """_summary_

        Args:
            num_frames (int): number of frames, i.e. batch size

        Returns:
            _type_: _description_
        """
        if self.preload_transitions:
            idxs = np.random.choice(
                self.preloaded_s.shape[0], size=num_frames)
            return self.preloaded_s[idxs]
        else:
            traj_idxs = self.weighted_traj_idx_sample_batch(num_frames)
            times = self.traj_time_sample_batch(traj_idxs)
            return self.get_full_frame_at_time_batch(traj_idxs, times)

    def blend_frame_pose(self, frame0, frame1, blend):
        """Linearly interpolate between two frames, including orientation.

        Args:
            frame0: First frame to be blended corresponds to (blend = 0).
            frame1: Second frame to be blended corresponds to (blend = 1).
            blend: Float between [0, 1], specifying the interpolation between
            the two frames.
        Returns:
            An interpolation of the two frames.
        """

        base_pos0, base_pos1 = self.get_base_pos(frame0), self.get_base_pos(frame1)
        base_rot0, base_rot1 = self.get_base_rot(frame0), self.get_base_rot(frame1)
        base_lin_vel0, base_lin_vel1 = self.get_base_lin_vel(frame0), self.get_base_lin_vel(frame1)
        base_ang_vel0, base_ang_vel1 = self.get_base_ang_vel(frame0), self.get_base_ang_vel(frame1)
        dof_pos0, dof_pos1 = self.get_dof_pos(frame0), self.get_dof_pos(frame1)
        dof_vel0, dof_vel1 = self.get_dof_vel(frame0), self.get_dof_vel(frame1)
        key_body_pos0, key_body_pos1 = self.get_key_body_pos_local(frame0), self.get_key_body_pos_local(frame1)

        blend_base_pos = self.slerp(base_pos0, base_pos1, blend)
        blend_base_rot = quat_slerp(base_rot0, base_rot1, blend)
        blend_base_rot = standardize_quaternion(blend_base_rot)
        blend_base_lin_vel = self.slerp(base_lin_vel0, base_lin_vel1, blend)
        blend_base_ang_vel = self.slerp(base_ang_vel0, base_ang_vel1, blend)
        blend_dof_pos = self.slerp(dof_pos0, dof_pos1, blend)
        blend_dof_vel = self.slerp(dof_vel0, dof_vel1, blend)
        blend_key_body_pos = self.slerp(key_body_pos0, key_body_pos1, blend)

        return torch.cat([
            blend_base_pos, blend_base_rot, blend_base_lin_vel, blend_base_ang_vel,
            blend_dof_pos, blend_dof_vel, blend_key_body_pos])

    def feed_forward_generator(self, num_mini_batch, mini_batch_size):
        """Generates a batch of AMP transitions."""
        for _ in range(num_mini_batch):
            if self.preload_transitions:
                idxs = np.random.choice(
                    self.preloaded_s.shape[0], size=mini_batch_size)
                # preloaded state comes from trajectory_full
                s = self.preloaded_s[idxs, self.base_lin_vel_start_idx:self.key_body_pos_end_idx]
                s_next = self.preloaded_s_next[idxs, self.base_lin_vel_start_idx:self.key_body_pos_end_idx]
            else:
                s, s_next = [], []
                traj_idxs = self.weighted_traj_idx_sample_batch(mini_batch_size)
                times = self.traj_time_sample_batch(traj_idxs)
                s = self.get_frame_at_time_batch(traj_idxs, times)
                s_next = self.get_frame_at_time_batch(traj_idxs, times + self.time_between_frames)
                
                # for traj_idx, frame_time in zip(traj_idxs, times):
                #     s.append(self.get_frame_at_time(traj_idx, frame_time))
                #     s_next.append(
                #         self.get_frame_at_time(
                #             traj_idx, frame_time + self.time_between_frames))
                
                # s = torch.vstack(s)
                # s_next = torch.vstack(s_next)
            yield s, s_next

    @property
    def observation_dim(self):
        """Size of AMP observations."""
        return self.trajectories[0].shape[1]

    @property
    def num_motions(self):
        return len(self.trajectory_names)
    
    def get_base_pos(self, pose):
        return pose[self.base_pos_start_idx:self.base_pos_end_idx]
    
    def get_base_pos_batch(self, poses):
        return poses[:, self.base_pos_start_idx:self.base_pos_end_idx]
    
    def get_base_rot(self, pose):
        return pose[self.base_rot_start_idx:self.base_rot_end_idx]
    
    def get_base_rot_batch(self, poses):
        return poses[:, self.base_rot_start_idx:self.base_rot_end_idx]
    
    def get_base_lin_vel(self, pose):
        return pose[self.base_lin_vel_start_idx:self.base_lin_vel_end_idx]
    
    def get_base_lin_vel_batch(self, poses):
        return poses[:, self.base_lin_vel_start_idx:self.base_lin_vel_end_idx]
    
    def get_base_ang_vel(self, pose):
        return pose[self.base_ang_vel_start_idx:self.base_ang_vel_end_idx]

    def get_base_ang_vel_batch(self, poses):
        return poses[:, self.base_ang_vel_start_idx:self.base_ang_vel_end_idx]

    def get_dof_pos(self, pose):
        return pose[self.dof_pos_start_idx:self.dof_pos_end_idx]

    def get_dof_pos_batch(self, poses):
        return poses[:, self.dof_pos_start_idx:self.dof_pos_end_idx]
    
    def get_dof_vel(self, pose):
        return pose[self.dof_vel_start_idx:self.dof_vel_end_idx]

    def get_dof_vel_batch(self, poses):
        return poses[:, self.dof_vel_start_idx:self.dof_vel_end_idx]  
    
    def get_key_body_pos_local(self, pose):
        return pose[self.key_body_pos_start_idx:self.key_body_pos_end_idx]

    def get_key_body_pos_local_batch(self, poses):
        return poses[:, self.key_body_pos_start_idx:self.key_body_pos_end_idx]
    
class MotionLoader:
    def __init__(self,
                 num_envs,
                 num_key_bodies,
                 motion_file,
                 device):
        """Loads motion data from a .pkl file and provides utilities for sampling frames for DeepMimic learning.
        
        Args:
            time_between_frames: Amount of time in seconds between transition.
            motion_file: file path to the motion data file (after {LEGGED_GYM_ROOT_DIR}/resources/reference_motion/).  
                The file should be a .pkl file containing a dictionary with keys "root_pos", "root_rot", "root_lin_vel", "root_ang_vel", "dof_pos", "dof_vel", and "key_body_pos_relative_to_base", 
                where the values are numpy arrays of shape (num_frames, feature_dim) corresponding to the trajectory of that feature over time.
            device: torch device to store the loaded trajectories.
        
        Attention:
            Only one motion file is supported in this loader.
        """
        self.device = device
        self.num_envs = num_envs
        
        # open the .pkl file and load the motion data
        # the motion data should be a dictionary with the following keys:
        # motion_data = {
        #     "fps": aligned_fps,
        #     "root_pos": root_pos.cpu().numpy(), # root link position in world frame
        #     "root_lin_vel": root_lin_vel.cpu().numpy(), # root link linear velocity in world frame
        #     "root_rot": root_rot.cpu().numpy(), # root link orientation as quaternion in world frame
        #     "root_euler": root_euler.cpu().numpy(), # root link orientation as euler angles in world frame
        #     "root_ang_vel": root_ang_vel.cpu().numpy(), # root link angular velocity in world frame
        #     "dof_pos": dof_pos.cpu().numpy(), # joint angles matching dof_names order
        #     "dof_vel": dof_vel.cpu().numpy(), # joint velocities matching dof_names order
        #     "key_body_pos_relative_to_base": key_body_pos_relative_to_base.cpu().numpy(),
        #           # key body point positions relative to the base in the world frame, shape [motion_length, num_key_bodies, 3]
        # }
        motion_file_dir = LEGGED_GYM_ROOT_DIR + "/resources/reference_motion/"
        motion_file_path = motion_file_dir + motion_file
        with open(motion_file_path, "rb") as f:
            motion_data = pickle.load(f)
        self.ref_base_pos = torch.from_numpy(motion_data["root_pos"]).to(self.device).float()
        if "root_lin_vel" in motion_data:
            self.ref_base_lin_vel = torch.from_numpy(motion_data["root_lin_vel"]).to(self.device).float()
        else:
            self.ref_base_lin_vel = torch.zeros_like(self.ref_base_pos)
        self.ref_base_quat = torch.from_numpy(motion_data["root_rot"]).to(self.device).float()
        if "root_ang_vel" in motion_data:
            self.ref_base_ang_vel = torch.from_numpy(motion_data["root_ang_vel"]).to(self.device).float()
        else:
            self.ref_base_ang_vel = torch.zeros_like(self.ref_base_pos)
        self.ref_dof_pos = torch.from_numpy(motion_data["dof_pos"]).to(self.device).float()
        if "dof_vel" in motion_data:
            self.ref_dof_vel = torch.from_numpy(motion_data["dof_vel"]).to(self.device).float()
        else:
            self.ref_dof_vel = torch.zeros_like(self.ref_dof_pos)
        if "key_body_pos_relative_to_base" in motion_data:
            self.ref_key_body_pos = torch.from_numpy(motion_data["key_body_pos_relative_to_base"]).to(self.device).float()
        else:
            self.ref_key_body_pos = torch.zeros(
                self.ref_base_pos.shape[0], 
                num_key_bodies,
                3, device=self.device, dtype=torch.float32
            )
        assert self.ref_base_pos.shape[0] == self.ref_base_quat.shape[0] == self.ref_base_lin_vel.shape[0] == self.ref_base_ang_vel.shape[0] == self.ref_dof_pos.shape[0] == self.ref_dof_vel.shape[0], "Reference motion data length mismatch among different features"
        
        self.trajectory_num_frames = motion_data["root_pos"].shape[0] # number of frames in the trajectory, calculated from the length of the root position data. Assuming all features have the same number of frames, we can use any of them to get the number of frames. This is used to determine how to sample frames from the trajectory based on time.
        self.max_index = int(self.trajectory_num_frames - 1) # maximum valid frame index, which is one less than the number of frames since indexing starts at 0. This is used to ensure that when we sample frames based on time, we don't go out of bounds.
        
        print(f"Loaded reference motion from {motion_file_path}, motion length: {self.trajectory_num_frames} frames")
    
        self.frame_index = torch.zeros(num_envs, device=self.device, dtype=torch.long)
        self.resample_frame_index(torch.arange(num_envs, device=self.device))
        
    def resample_frame_index(self, env_ids=None):
        """Resample frame index for the given env ids."""
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
        self.frame_index[env_ids] = torch.randint(0, self.trajectory_num_frames, (len(env_ids),), device=self.device) # offset 1 frame to avoid idx out of bound
    
    def step_frame_index(self):
        """Step frame index for all envs"""
        self.frame_index += 1
        # reset to 0 if exceeds max index
        time_out_env_ids = (self.frame_index > self.max_index).nonzero(as_tuple=False).flatten()
        self.frame_index[time_out_env_ids] = 0
        return time_out_env_ids
    
    def get_ref_base_pos(self, env_ids=None):
        """Get reference base position for the given env ids."""
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
        return self.ref_base_pos[self.frame_index[env_ids]]
    
    def get_ref_base_pos_at_idx(self, idx):
        """Get reference base position at the specified index."""
        return self.ref_base_pos[idx]

    def get_ref_base_quat(self, env_ids=None):
        """Get reference base orientation as quaternion for the given env ids."""
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
        return self.ref_base_quat[self.frame_index[env_ids]]
    
    def get_ref_base_quat_at_idx(self, idx):
        """Get reference base orientation as quaternion at the specified index."""
        return self.ref_base_quat[idx]
    
    def get_ref_base_lin_vel(self, env_ids=None):
        """Get reference base linear velocity for the given env ids."""
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
        return self.ref_base_lin_vel[self.frame_index[env_ids]]
    
    def get_ref_base_lin_vel_at_idx(self, idx):
        """Get reference base linear velocity at the specified index."""
        return self.ref_base_lin_vel[idx]

    def get_ref_base_ang_vel(self, env_ids=None):
        """Get reference base angular velocity for the given env ids."""
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
        return self.ref_base_ang_vel[self.frame_index[env_ids]]
    
    def get_ref_base_ang_vel_at_idx(self, idx):
        """Get reference base angular velocity at the specified index."""
        return self.ref_base_ang_vel[idx]
    
    def get_ref_dof_pos(self, env_ids=None):
        """Get reference joint positions for the given env ids."""
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
        return self.ref_dof_pos[self.frame_index[env_ids]]
    
    def get_ref_dof_pos_at_idx(self, idx):
        """Get reference joint positions at the specified index."""
        return self.ref_dof_pos[idx]

    def get_ref_dof_vel(self, env_ids=None):
        """Get reference joint velocities for the given env ids."""
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
        return self.ref_dof_vel[self.frame_index[env_ids]]
    
    def get_ref_dof_vel_at_idx(self, idx):
        """Get reference joint velocities at the specified index."""
        return self.ref_dof_vel[idx]

    def get_ref_key_body_pos(self, env_ids=None):
        """Get reference key body positions for the given env ids."""
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
        return self.ref_key_body_pos[self.frame_index[env_ids]]
    
    def get_ref_key_body_pos_at_idx(self, idx):
        """Get reference key body positions at the specified time."""
        return self.ref_key_body_pos[idx]