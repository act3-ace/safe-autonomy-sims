import argparse
import scipy
from matplotlib import pyplot as plt
import numpy as np

from safe_autonomy_dynamics.cwh import M_DEFAULT, N_DEFAULT, generate_cwh_matrices

from scripts.base_rta_test import RTAExperiment, main


class TestController():
    def __init__(self):
        self.u_max = 1

        A, B = generate_cwh_matrices(M_DEFAULT, N_DEFAULT, mode="3d")

        # Specify LQR gains
        Q = np.eye(6) * 0.05  # State cost
        R = np.eye(3) * 1000  # Control cost

        # Solve ARE
        Xare = np.matrix(scipy.linalg.solve_continuous_are(A, B, Q, R))
        self.Klqr = np.array(-scipy.linalg.inv(R)*(B.T*Xare))

    def compute_feedback_control(self, x):
        u = self.Klqr @ x
        return np.clip(u, -self.u_max, self.u_max)

    def make_action(self, agent_name, x_thrust, y_thrust, z_thrust):
        action = {
            agent_name: {
                "RTAModule":
                    (
                        {'x_thrust': np.array([x_thrust], dtype=np.float32)},
                        {'y_thrust': np.array([y_thrust], dtype=np.float32)},
                        {'z_thrust': np.array([z_thrust], dtype=np.float32)},
                    )
            }
        }
        return action


class DockingRTAExperiment(RTAExperiment):
    def run_experiment(self, args: argparse.Namespace) -> None:
        args, env = self.setup_experiment(args)

        obs = env.reset()

        agent = args.agent_config[0].name
        data = []
        controller = TestController()
        done = False
        i = 0

        while not done:

            position = obs[agent]["ObserveSensor_Sensor_Position"]["direct_observation"]
            velocity = obs[agent]["ObserveSensor_Sensor_Velocity"]["direct_observation"]
            state = np.concatenate((position, velocity))

            u_des = controller.compute_feedback_control(state)

            action = controller.make_action(agent, *u_des)
            obs, rewards, dones, info = env.step(action)

            intervening = info[agent]['RTAModule']['intervening']
            control_actual = info[agent]['RTAModule']['actual control']

            done = dones['__all__']

            step_data = {
                'state': state,
                'intervening': intervening,
                'control': control_actual,
            }
            print('completed step=', i)
            data.append(step_data)
            i += 1

        self.plot_pos_vel(data)

    def plot_pos_vel(self, data):

        states = np.empty([len(data), 6])
        control = np.empty([len(data), 3])
        intervening = np.empty([len(data)])
        for i in range(len(data)):
            states[i, :] = data[i]['state']
            control[i, :] = data[i]['control']
            intervening[i] = data[i]['intervening']

        fig = plt.figure()
        fig.set_figwidth(15)
        fig.set_figheight(10)
        ax1 = fig.add_subplot(231, projection='3d')
        ax2 = fig.add_subplot(232)
        ax3 = fig.add_subplot(233)
        ax4 = fig.add_subplot(234)
        ax5 = fig.add_subplot(235)
        ax6 = fig.add_subplot(236)

        max = np.max(np.abs(states[:, 0:3]))*1.1
        RTAon = np.ma.masked_where(intervening != 1, states[:, 1])
        ax1.plot(0, 0, 'kx')
        ax1.plot(states[0, 0], states[0, 1], states[0, 2], 'r+')
        ax1.plot(states[:, 0], states[:, 1], states[:, 2], 'b')
        ax1.plot(states[:, 0], RTAon, states[:, 2], 'c')
        ax1.set_xlim([-max, max])
        ax1.set_ylim([-max, max])
        ax1.set_zlim([-max, max])
        ax1.set_xlabel('x')
        ax1.set_ylabel('y')
        ax1.set_zlabel('y')
        ax1.set_title('Trajectory')
        ax1.grid(True)

        v = np.empty([len(states), 2])
        for i in range(len(states)):
            v[i, :] = [np.linalg.norm(states[i, 0:3]), np.linalg.norm(states[i, 3:6])]
        RTAon = np.ma.masked_where(intervening != 1, v[:, 1])
        ax2.plot([0, 15000], [0.2, 0.2+4*0.001027*15000], 'g--')
        ax2.plot(v[:, 0], v[:, 1], 'b')
        ax2.plot(v[:, 0], RTAon, 'c')
        ax2.set_xlim([0, np.max(v[:, 0])*1.1])
        ax2.set_ylim([0, np.max(v[:, 1])*1.1])
        ax2.set_xlabel('Relative Position')
        ax2.set_ylabel('Relative Velocity')
        ax2.set_title('Distance Dependent Speed Limit')
        ax2.grid(True)

        ax3.plot(0, 0, 'kx')
        ax3.plot(0, 0, 'r+')
        ax3.plot(0, 0, 'g')
        ax3.plot(0, 0, 'b')
        ax3.plot(0, 0, 'c')
        ax3.legend(['Chief', 'Deputy Initial Position', 'Constraint', 'RTA Not Intervening', 'RTA Intervening'])
        ax3.axis('off')
        ax3.set_xlim([1, 2])
        ax3.set_ylim([1, 2])

        RTAonx = np.ma.masked_where(intervening != 1, states[:, 3])
        RTAony = np.ma.masked_where(intervening != 1, states[:, 4])
        RTAonz = np.ma.masked_where(intervening != 1, states[:, 5])
        ax4.plot([0, len(states)*1.1], [10, 10], 'g--')
        ax4.plot([0, len(states)*1.1], [-10, -10], 'g--')
        ax4.plot(range(len(states)), states[:, 3], 'b')
        ax4.plot(range(len(states)), RTAonx, 'c')
        ax4.plot(range(len(states)), states[:, 4], 'r')
        ax4.plot(range(len(states)), RTAony, 'tab:orange')
        ax4.plot(range(len(states)), states[:, 5], 'tab:brown')
        ax4.plot(range(len(states)), RTAonz, 'tab:green')
        ax4.set_xlim([0, len(states)*1.1])
        ax4.set_xlabel('Time Steps')
        ax4.set_ylabel('Velocity')
        ax4.set_title('Max Velocity Constraint')
        ax4.grid(True)

        RTAonx = np.ma.masked_where(intervening != 1, control[:, 0])
        RTAony = np.ma.masked_where(intervening != 1, control[:, 1])
        RTAonz = np.ma.masked_where(intervening != 1, control[:, 2])
        ax5.plot([0, len(control)*1.1], [1, 1], 'g--')
        ax5.plot([0, len(control)*1.1], [-1, -1], 'g--')
        ax5.plot(range(len(control)), control[:, 0], 'b')
        ax5.plot(range(len(control)), RTAonx, 'c')
        ax5.plot(range(len(control)), control[:, 1], 'r')
        ax5.plot(range(len(control)), RTAony, 'tab:orange')
        ax5.plot(range(len(control)), control[:, 2], 'tab:brown')
        ax5.plot(range(len(control)), RTAonz, 'tab:green')
        ax5.set_xlim([0, len(control)*1.1])
        ax5.set_xlabel('Time Steps')
        ax5.set_ylabel('Force')
        ax5.set_title('Actions')
        ax5.grid(True)

        ax6.plot(0, 0, 'g--')
        ax6.plot(0, 0, 'b')
        ax6.plot(0, 0, 'c')
        ax6.plot(0, 0, 'r')
        ax6.plot(0, 0, 'tab:orange')
        ax6.plot(0, 0, 'tab:brown')
        ax6.plot(0, 0, 'tab:green')
        ax6.legend(['Constraint', 'vx/Fx: RTA Not Intervening', 'vx/Fx: RTA Intervening', 'vy/Fy: RTA Not Intervening',
                    'vy/Fy: RTA Intervening', 'vz/Fz: RTA Not Intervening', 'vz/Fz: RTA Intervening'])
        ax6.axis('off')
        ax6.set_xlim([1, 2])
        ax6.set_ylim([1, 2])

        plt.show()
        # plt.savefig('rta_test.png')

    def plot_constraint_line(self, ax, x, y, border_linewidth=10, color='r', border_alpha=0.25):
        ax.plot(x, y, color, linewidth=border_linewidth, alpha=border_alpha)
        ax.plot(x, y, color)


if __name__ == "__main__":
    main()
