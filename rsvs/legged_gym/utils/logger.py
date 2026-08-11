from multiprocessing import Process, Value
from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import xlsxwriter


# Fix font issues
plt.rcParams.update({
    'font.size': 10,
    'axes.titlesize': 10,
    'axes.labelsize': 9,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'legend.fontsize': 8,
    'figure.titlesize': 12
})

EXCEL_FILENAME = "/home/lupinjia/Documents/2025/genesis_test/20250724_go2_deploy_step_gait.xlsx"


class Logger:
    def __init__(self, dt):
        self.state_log = defaultdict(list)
        self.rew_log = defaultdict(list)
        self.dt = dt
        self.num_episodes = 0
        self.plot_process = None

    def log_state(self, key, value):
        self.state_log[key].append(value)

    def log_states(self, dict):
        for key, value in dict.items():
            self.log_state(key, value)

    def log_rewards(self, dict, num_episodes):
        for key, value in dict.items():
            if 'rew' in key:
                self.rew_log[key].append(value.item() * num_episodes)
        self.num_episodes += num_episodes

    def reset(self):
        self.state_log.clear()
        self.rew_log.clear()

    def plot_states(self):
        self.plot_process = Process(target=self._plot)
        self.plot_process.start()

    def _plot(self):
        # Set matplotlib parameters to avoid font issues
        plt.rcParams['font.family'] = 'DejaVu Sans'
        plt.rcParams['font.size'] = 8

        nb_rows = 3
        nb_cols = 3
        fig = plt.figure(figsize=(12, 9), dpi=80)
        axs = fig.subplots(nb_rows, nb_cols)

        # Calculate time array
        time = None
        for key, value in self.state_log.items():
            if value:
                time = np.linspace(0, len(value)*self.dt, len(value))
                break

        if time is None:
            print("No data to plot")
            return

        log = self.state_log

        # plot joint targets and measured positions
        a = axs[1, 0]
        if log["dof_pos"]:
            a.plot(time, log["dof_pos"], label='measured')
        if log["dof_pos_target"]:
            a.plot(time, log["dof_pos_target"], label='target')
        a.set(xlabel='time [s]', ylabel='Position [rad]', title='DOF Position')
        if log["dof_pos"] or log["dof_pos_target"]:
            a.legend()

        # plot joint velocity
        a = axs[1, 1]
        if log["dof_vel"]:
            a.plot(time, log["dof_vel"], label='measured')
        if log["dof_vel_target"]:
            a.plot(time, log["dof_vel_target"], label='target')
        a.set(xlabel='time [s]', ylabel='Velocity [rad/s]',
              title='Joint Velocity')
        if log["dof_vel"] or log["dof_vel_target"]:
            a.legend()

        # plot base vel x
        a = axs[0, 0]
        if log["base_vel_x"]:
            a.plot(time, log["base_vel_x"], label='measured')
        if log["command_x"]:
            a.plot(time, log["command_x"], label='commanded')
        a.set(xlabel='time [s]', ylabel='base lin vel [m/s]',
              title='Base velocity x')
        if log["base_vel_x"] or log["command_x"]:
            a.legend()

        # plot base vel y
        a = axs[0, 1]
        if log["base_vel_y"]:
            a.plot(time, log["base_vel_y"], label='measured')
        if log["command_y"]:
            a.plot(time, log["command_y"], label='commanded')
        a.set(xlabel='time [s]', ylabel='base lin vel [m/s]',
              title='Base velocity y')
        if log["base_vel_y"] or log["command_y"]:
            a.legend()

        # plot base vel yaw
        a = axs[0, 2]
        if log["base_vel_yaw"]:
            a.plot(time, log["base_vel_yaw"], label='measured')
        if log["command_yaw"]:
            a.plot(time, log["command_yaw"], label='commanded')
        a.set(xlabel='time [s]', ylabel='base ang vel [rad/s]',
              title='Base velocity yaw')
        if log["base_vel_yaw"] or log["command_yaw"]:
            a.legend()

        # plot base vel z
        a = axs[1, 2]
        if log["base_vel_z"]:
            a.plot(time, log["base_vel_z"], label='measured')
        a.set(xlabel='time [s]', ylabel='base lin vel [m/s]',
              title='Base velocity z')
        if log["base_vel_z"]:
            a.legend()

        # plot contact forces
        a = axs[2, 0]
        if log["contact_forces_z"]:
            forces = np.array(log["contact_forces_z"])
            for i in range(forces.shape[1]):
                a.plot(time, forces[:, i], label=f'force {i}')
            a.legend()
        a.set(xlabel='time [s]', ylabel='Forces z [N]',
              title='Vertical Contact forces')

        # plot torque/vel curves
        a = axs[2, 1]
        if log["dof_vel"] != [] and log["dof_torque"] != []:
            a.plot(log["dof_vel"], log["dof_torque"], 'x', label='measured')
            a.legend()
        a.set(xlabel='Joint vel [rad/s]',
              ylabel='Joint Torque [Nm]', title='Torque/velocity curves')

        # plot torques
        a = axs[2, 2]
        if log["dof_torque"] != []:
            a.plot(time, log["dof_torque"], label='measured')
            a.legend()
        a.set(xlabel='time [s]', ylabel='Joint Torque [Nm]', title='Torque')

        plt.tight_layout()
        # Save to file instead of show() for non-interactive backend
        plt.savefig('./plot.png', dpi=100, bbox_inches='tight')
        plt.close(fig)
        print("Plots saved to scripts/plot.png")

    def print_rewards(self):
        print("Average rewards per second:")
        for key, values in self.rew_log.items():
            mean = np.sum(np.array(values)) / self.num_episodes
            print(f" - {key}: {mean}")
        print(f"Total number of episodes: {self.num_episodes}")
        
    def __del__(self):
        if self.plot_process is not None:
            self.plot_process.kill()

class QuadLogger(Logger):
    def __init__(self, dt):
        super().__init__(dt)
    
    def set_header_of_xlsx(self):
        self.workbook = xlsxwriter.Workbook(EXCEL_FILENAME)
        self.worksheet = self.workbook.add_worksheet()
        label = list(self.state_log.keys())
        for i in range(len(label)):
            self.worksheet.write(0, i, label[i])
    
    def save_data_to_xlsx(self):
        '''
        save the data to a excel file
        '''
        # set header
        self.set_header_of_xlsx()
        # get the first key, to get the length of the data
        first_key = list(self.state_log.keys())[0]
        for row in range(len(self.state_log[first_key])):
            for col, key in enumerate(self.state_log.keys()):
                self.worksheet.write(1+row, col, self.state_log[key][row])
        self.workbook.close()
        print("xlsx file created and filled!")

    def _plot(self):
        # Set matplotlib parameters to avoid font issues
        plt.rcParams['font.family'] = 'DejaVu Sans'
        plt.rcParams['font.size'] = 8

        nb_rows = 3
        nb_cols = 4
        fig = plt.figure(figsize=(12, 9), dpi=80)
        axs = fig.subplots(nb_rows, nb_cols)

        # Calculate time array
        time = None
        for key, value in self.state_log.items():
            if value:
                time = np.linspace(0, len(value)*self.dt, len(value))
                break

        if time is None:
            print("No data to plot")
            return

        log = self.state_log

        # plot base vel x
        a = axs[0, 0]
        if log["base_vel_x"]:
            a.plot(time, log["base_vel_x"], label='measured')
        if log["command_x"]:
            a.plot(time, log["command_x"], label='commanded')
        a.set(xlabel='time [s]', ylabel='base lin vel [m/s]',
              title='Base velocity x')
        if log["base_vel_x"] or log["command_x"]:
            a.legend()

        # plot base vel y
        a = axs[0, 1]
        if log["base_vel_y"]:
            a.plot(time, log["base_vel_y"], label='measured')
        if log["command_y"]:
            a.plot(time, log["command_y"], label='commanded')
        a.set(xlabel='time [s]', ylabel='base lin vel [m/s]',
              title='Base velocity y')
        if log["base_vel_y"] or log["command_y"]:
            a.legend()

        # plot base vel yaw
        a = axs[0, 2]
        if log["base_vel_yaw"]:
            a.plot(time, log["base_vel_yaw"], label='measured')
        if log["command_yaw"]:
            a.plot(time, log["command_yaw"], label='commanded')
        a.set(xlabel='time [s]', ylabel='base ang vel [rad/s]',
              title='Base velocity yaw')
        if log["base_vel_yaw"] or log["command_yaw"]:
            a.legend()

        # plot exp_C_frc
        a = axs[1, 0]
        if log["exp_C_frc_fl"]:
            a.plot(time, log["exp_C_frc_fl"], label='exp_C_frc_fl')
            a.legend()     
        a = axs[1, 1]
        if log["exp_C_frc_fr"]:
            a.plot(time, log["exp_C_frc_fr"], label='exp_C_frc_fr')
            a.legend()
        a = axs[1, 2]
        if log["exp_C_frc_rl"]:
            a.plot(time, log["exp_C_frc_rl"], label='exp_C_frc_rl')
            a.legend()
        a = axs[1, 3]
        if log["exp_C_frc_rr"]:
            a.plot(time, log["exp_C_frc_rr"], label='exp_C_frc_rr')
            a.legend()
        
        # plot foot contact forces
        a = axs[2, 0]
        if log["contact_forces_fl"]:
            a.plot(time, log["contact_forces_fl"], label='contact_forces_fl')
            a.legend()
        a = axs[2, 1]
        if log["contact_forces_fr"]:
            a.plot(time, log["contact_forces_fr"], label='contact_forces_fr')
            a.legend()
        a = axs[2, 2]
        if log["contact_forces_rl"]:
            a.plot(time, log["contact_forces_rl"], label='contact_forces_rl')
            a.legend()
        a = axs[2, 3]
        if log["contact_forces_rr"]:
            a.plot(time, log["contact_forces_rr"], label='contact_forces_rr')
            a.legend()

        plt.tight_layout()
        # Save to file instead of show() for non-interactive backend
        plt.savefig('./plot.png', dpi=100, bbox_inches='tight')
        plt.close(fig)
        print("Plots saved to scripts/plot.png")
