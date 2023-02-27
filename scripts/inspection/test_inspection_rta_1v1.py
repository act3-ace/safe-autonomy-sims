import argparse
from matplotlib import pyplot as plt
import numpy as np


from scripts.base_rta_test import RTAExperiment, main
from scripts.test_cwh_rta_final import TestController


class Inspection1v1RTAExperiment(RTAExperiment):
    def run_experiment(self, args: argparse.Namespace) -> None:
        args, env = self.setup_experiment(args)

        obs = env.reset()

        agent = args.agent_config[0][0]
        data = []
        controller = TestController()
        done = False
        i = 0

        while not done:

            position = obs[agent]["ObserveSensor_Sensor_Position"]["direct_observation"]*100
            velocity = obs[agent]["ObserveSensor_Sensor_Velocity"]["direct_observation"]*0.5
            state = np.concatenate((position, velocity))

            u_des = controller.compute_feedback_control(state)

            action = controller.make_action(agent, *u_des)
            obs, rewards, dones, info = env.step(action)

            intervening = info['blue0']['RTAModule']['intervening']
            control_actual = info['blue0']['RTAModule']['actual control']

            done = dones['__all__']
            if i >= 1000:
                done = True

            step_data = {
                'state': state,
                'intervening': intervening,
                'control': control_actual,
            }
            print('completed step=', i)
            data.append(step_data)
            i += 1

        self.make_plots(data)

    def make_plots(self, data):
        v0 = 0.2
        v0_distance = 10
        v1 = 2 * 0.001027
        collision_radius = 10
        u_max = 1
        sensor_fov = 60 * np.pi / 180
        vel_limit = 1
        r_max = 1000

        array = np.empty([len(data), 6])
        control = np.empty([len(data), 3])
        intervening = np.empty([len(data)])
        for i in range(len(data)):
            array[i, :] = data[i]['state']
            control[i, :] = data[i]['control']
            intervening[i] = data[i]['intervening']

        fig = plt.figure(figsize=(15, 15))
        ax1 = fig.add_subplot(331, projection='3d')
        ax2 = fig.add_subplot(332)
        ax3 = fig.add_subplot(333)
        ax5 = fig.add_subplot(334)
        ax6 = fig.add_subplot(335)
        ax8 = fig.add_subplot(336)
        ax9 = fig.add_subplot(337)
        lw = 2
        
        ax1.plot(array[:, 0], array[:, 1], array[:, 2], linewidth=lw)
        max = np.max(np.abs(array[:, 0:3]))*1.1
        ax1.plot(0, 0, 0, 'k*', markersize=15)
        ax1.set_xlabel(r'$x$ [m]')
        ax1.set_ylabel(r'$y$ [m]')
        ax1.set_zlabel(r'$z$ [m]')
        ax1.set_xlim([-max, max])
        ax1.set_ylim([-max, max])
        ax1.set_zlim([-max, max])
        ax1.set_box_aspect((1,1,1))
        ax1.grid(True)

        v = np.empty([len(array), 2])
        for j in range(len(array)):
            v[j, :] = [np.linalg.norm(array[j, 0:3]), np.linalg.norm(array[j, 3:6])]
        ax2.plot(range(len(array)), v[:, 0], linewidth=lw)
        xmax = len(array)*1.1
        ymax = np.max(v[:, 0])*1.1
        ax2.fill_between([0, xmax], [collision_radius, collision_radius], [ymax, ymax], color=(244/255, 249/255, 241/255))
        ax2.fill_between([0, xmax], [0, 0], [collision_radius, collision_radius], color=(255/255, 239/255, 239/255))
        ax2.plot([0, xmax], [collision_radius, collision_radius], 'k--', linewidth=lw)
        ax2.set_xlim([0, xmax])
        ax2.set_xlabel('Time [s]')
        ax2.set_ylabel(r'Relative Dist. ($\vert \vert \mathbf{p} \vert \vert_2$) [m]')
        ax2.set_yscale('log')
        ax2.set_ylim([6, ymax])
        ax2.grid(True)

        xmax = np.max(v[:, 0])*1.1
        ymax = np.max(v[:, 1])*1.1
        ax3.plot(v[:, 0], v[:, 1], linewidth=lw)
        ax3.fill_between([v0_distance, xmax], [v0, v0 + v1*(xmax-v0_distance)], [ymax, ymax], color=(255/255, 239/255, 239/255))
        ax3.fill_between([v0_distance, xmax], [0, 0], [v0, v0 + v1*(xmax-v0_distance)], color=(244/255, 249/255, 241/255))
        ax3.fill_between([0, v0_distance], [0, 0], [ymax, ymax], color=(255/255, 239/255, 239/255)) # red
        ax3.plot([v0_distance, xmax], [v0, v0 + v1*(xmax-v0_distance)], 'k--', linewidth=lw)
        ax3.plot([v0_distance, v0_distance], [0, v0], 'k--', linewidth=lw)
        ax3.set_xlim([0, xmax])
        ax3.set_ylim([0, ymax])
        ax3.set_xlabel(r'Relative Dist. ($\vert \vert \mathbf{p}  \vert \vert_2$) [m]')
        ax3.set_ylabel(r'Relative Vel. ($\vert \vert \mathbf{v}  \vert \vert_2$) [m/s]')
        ax3.grid(True)

        th = sensor_fov/2*180/np.pi
        xmax = len(array)*1.1
        h = np.zeros(len(array))
        for j in range(len(array)):
            # r_s_hat = np.array([np.cos(array[j, 6]), np.sin(array[j, 6]), 0.])
            r_s_hat = np.array([np.cos(0), np.sin(0), 0.])
            r_b_hat = -array[j, 0:3]/np.linalg.norm(array[j, 0:3])
            h[j] = np.arccos(np.dot(r_s_hat, r_b_hat))*180/np.pi
        ymax = np.max(h)*1.1
        ax5.plot(range(len(array)), h, linewidth=lw)
        ax5.fill_between([0, xmax], [th, th], [ymax, ymax], color=(244/255, 249/255, 241/255))
        ax5.fill_between([0, xmax], [0, 0], [th, th], color=(255/255, 239/255, 239/255))
        ax5.plot([0, xmax], [th, th], 'k--', linewidth=lw)
        ax5.set_xlim([0, xmax])
        ax5.set_ylim([0, ymax])
        ax5.set_xlabel('Time [s]')
        ax5.set_ylabel(r'Angle to Sun ($\theta_{EZ}}$) [degrees]')
        ax5.grid(True)

        ax6.plot(range(len(array)), v[:, 0], linewidth=lw)
        xmax = len(array)*1.1
        ymax = np.maximum(np.max(v[:, 0])*1.1, r_max*1.1)
        ax6.fill_between([0, xmax], [r_max, r_max], [ymax, ymax], color=(255/255, 239/255, 239/255))
        ax6.fill_between([0, xmax], [0, 0], [r_max, r_max], color=(244/255, 249/255, 241/255))
        ax6.plot([0, xmax], [r_max, r_max], 'k--', linewidth=lw)
        ax6.set_xlim([0, xmax])
        ax6.set_xlabel('Time [s]')
        ax6.set_ylabel(r'Relative Dist. ($\vert \vert \mathbf{p} \vert \vert_2$) [m]')
        ax6.set_ylim([0, ymax])
        ax6.grid(True)

        xmax = len(array)*1.1
        ymax = vel_limit*1.2
        ax8.plot(range(len(array)), array[:, 3], linewidth=lw, label=r'$\dot{x}$')
        ax8.plot(range(len(array)), array[:, 4], linewidth=lw, label=r'$\dot{y}$')
        ax8.plot(range(len(array)), array[:, 5], linewidth=lw, label=r'$\dot{z}$')
        ax8.fill_between([0, xmax], [vel_limit, vel_limit], [ymax, ymax], color=(255/255, 239/255, 239/255))
        ax8.fill_between([0, xmax], [-ymax, -ymax], [vel_limit, vel_limit], color=(255/255, 239/255, 239/255))
        ax8.fill_between([0, xmax], [-vel_limit, -vel_limit], [vel_limit, vel_limit], color=(244/255, 249/255, 241/255))
        ax8.plot([0, xmax], [vel_limit, vel_limit], 'k--', linewidth=lw)
        ax8.plot([0, xmax], [-vel_limit, -vel_limit], 'k--', linewidth=lw)
        ax8.set_xlim([0, xmax])
        ax8.set_ylim([-ymax, ymax])
        ax8.set_xlabel('Time [s]')
        ax8.set_ylabel(r'Velocity ($\mathbf{v}$) [m/s]')
        ax8.grid(True)
        ax8.legend()

        xmax = len(array)*1.1
        ymax = u_max*1.2
        ax9.plot(range(len(control)), control[:, 0], linewidth=lw, label=r'$F_x$')
        ax9.plot(range(len(control)), control[:, 1], linewidth=lw, label=r'$F_y$')
        ax9.plot(range(len(control)), control[:, 2], linewidth=lw, label=r'$F_z$')
        ax9.fill_between([0, xmax], [u_max, u_max], [ymax, ymax], color=(255/255, 239/255, 239/255))
        ax9.fill_between([0, xmax], [-ymax, -ymax], [u_max, u_max], color=(255/255, 239/255, 239/255))
        ax9.fill_between([0, xmax], [-u_max, -u_max], [u_max, u_max], color=(244/255, 249/255, 241/255))
        ax9.plot([0, xmax], [u_max, u_max], 'k--', linewidth=lw)
        ax9.plot([0, xmax], [-u_max, -u_max], 'k--', linewidth=lw)
        ax9.set_xlim([0, xmax])
        ax9.set_ylim([-ymax, ymax])
        ax9.set_xlabel('Time [s]')
        ax9.set_ylabel(r'$\mathbf{u}$ [N]')
        ax9.grid(True)
        ax9.legend()

        ax1.set_title('Trajectory')
        ax2.set_title('Safe Separation')
        ax3.set_title('Dynamic Speed Constraint')
        ax5.set_title('Keep Out Zone (Sun Avoidance)')
        ax6.set_title('Keep In Zone')
        ax8.set_title('Velocity Limits')
        ax9.set_title('Actuation Constraints')
        fig.tight_layout(h_pad=2)

        plt.show()

    def plot_constraint_line(self, ax, x, y, border_linewidth=10, color='r', border_alpha=0.25):
        ax.plot(x, y, color, linewidth=border_linewidth, alpha=border_alpha)
        ax.plot(x, y, color)


if __name__ == "__main__":
    main()
