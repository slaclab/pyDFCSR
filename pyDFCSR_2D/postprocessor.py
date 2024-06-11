import numpy as np
import matplotlib.pyplot as plt
from tools import full_path, find_nearest_ind, plot_2D_contour
import h5py
from matplotlib import cm

class DFCSR_postprocessor():
    """
    load output files from CSR class and make plots
    """

    def __init__(self, run_name, work_dir='./output/'):

        path = full_path(work_dir)

        self.wake_filename = f'{path}/{run_name}-wakes.h5'
        self.particle_filename = f'{path}/{run_name}-particles.h5'
        self.statistics_filename = f'{path}/{run_name}-statistics.h5'

       

    def parse_all_wakes(self):
        
        with h5py.File(self.wake_filename, "r") as f:
            self.charge_list = []
            self.energy_list = []
            self.gamma_list = []
            self.step_list = []
            self.element_list = []
            self.position_list = []

            for step in f.keys():
                self.energy_list.append(f[step].attrs['beam_energy'])
                self.charge_list.append(f[step].attrs['charge'])
                self.gamma_list.append(f[step].attrs['mean_gamma'])
                self.step_list.append(f[step].attrs['step'])
                self.position_list.append(f[step].attrs['position'])
                self.element_list.append(f[step].attrs['element'])
        

        with h5py.File(self.wake_filename, "r") as f:
            self.long_wake_list = []
            self.trans_wake_list = []
            self.long_unit_list = []
            self.trans_unit_list = []
            self.x_grids_list = []
            self.z_grids_list = []
            self.position_list = []

            for step in f.keys():
                print("Parsing wakes at step ", step)

                position = f[step].attrs['position']

                dE_dct = np.array(f[step]['longitudinal']['dE_dct'])
                self.long_unit_list.append(f[step]['longitudinal'].attrs['unit'])

                x_grids = np.array(f[step]['longitudinal']['x_grids']).reshape(dE_dct.shape)
                z_grids = np.array(f[step]['longitudinal']['z_grids']).reshape(dE_dct.shape)

                xkicks = np.array(f[step]['transverse']['xkicks'])
                self.trans_unit_list.append(f[step]['transverse'].attrs['unit'])

                self.long_wake_list.append(dE_dct)
                self.trans_wake_list.append(xkicks)
                self.x_grids_list.append(x_grids)
                self.z_grids_list.append(z_grids)
                self.positions_list.append(position)

    def parse_all_particles(self):

        with h5py.File(self.particle_filename, "r") as f:
            self.particles_list = []
            self.position_list = []

            for step in f.keys():
                print("Parsing particles at step ", step)

                position = f[step].attrs['position']
                x = np.array(f[step]['particles']['x'])
                y = np.array(f[step]['particles']['y'])
                xp = np.array(f[step]['particles']['xp'])
                yp = np.array(f[step]['particles']['yp'])
                z = np.array(f[step]['particles']['z'])
                delta = np.array(f[step]['particles']['delta'])

                self.particles_list.append(np.vstack((x, xp, y, yp, z, delta)).T)
                self.position_list.append(position)

    # 'Cx', 'Cxp', 'R51', 'R52', 'R56', 'alphaX', 'alphaX_beam',
    # 'alphaX_minus_dispersion', 'betaX', 'betaX_beam',
    # 'betaX_minus_dispersion', 'coords', 'etaX', 'etaXp',
    # 'gemitX', 'gemitX_minus_dispersion', 'n_vec', 'sigE',
    # 'sigX', 'sigZ', 'slope', 'step_positions', 'tau_vec'

    def get_statistics(self, key=None, show_plot = True):
        with h5py.File(self.statistics_filename, "r") as f:
            x = np.array(f['step_positions'])
            y = np.array(f[key])

        if show_plot:
            plt.figure()
            plt.plot(x, y)
            plt.xlabel('positions (m)')
            plt.ylabel(f'{key} (m)')
            plt.show()

        return x, y

    def get_wakes(self, s, show_plot = True):
        
        with h5py.File(self.wake_filename, "r") as f:
            self.charge_list = []
            self.energy_list = []
            self.gamma_list = []
            self.step_list = []
            self.element_list = []
            self.position_list = []

            for step in f.keys():
                self.energy_list.append(f[step].attrs['beam_energy'])
                self.charge_list.append(f[step].attrs['charge'])
                self.gamma_list.append(f[step].attrs['mean_gamma'])
                self.step_list.append(f[step].attrs['step'])
                self.position_list.append(f[step].attrs['position'])
                self.element_list.append(f[step].attrs['element'])
        
        ind = find_nearest_ind(self.position_list, s)

        print("plot longitudinal wakes at nearest point s  = {} m, step count {}".format(self.position_list[ind],
                                                                                         self.step_list[ind]))

        step = "step_" + str(self.step_list[ind])

        with h5py.File(self.wake_filename, "r") as f:
            print("ebeam energy {}".format(f[step].attrs['beam_energy']))
            dE_dct = np.array(f[step]['longitudinal']['dE_dct'])
            unit = f[step]['longitudinal'].attrs['unit']

            x_grids = np.array(f[step]['longitudinal']['x_grids']).reshape(dE_dct.shape)
            z_grids = np.array(f[step]['longitudinal']['z_grids']).reshape(dE_dct.shape)

            xkicks = np.array(f[step]['transverse']['xkicks'])

        if show_plot:

            fig, ax = plt.subplots(subplot_kw={"projection": "3d"}, figsize=(10, 8))
            surf = ax.plot_surface(x_grids * 1e6, z_grids * 1e6, dE_dct, cmap=cm.coolwarm,
                                   linewidth=0, antialiased=False)
            ax.set_xlabel('x (um)')
            ax.set_ylabel('z (um)')
            ax.set_zlabel("Longitudinal ({})".format(unit))
            fig.colorbar(surf, shrink=0.5, aspect=5)
            plt.show()

            fig, ax = plt.subplots(subplot_kw={"projection": "3d"}, figsize=(10, 8))
            surf = ax.plot_surface(x_grids * 1e6, z_grids * 1e6, xkicks, cmap=cm.coolwarm,
                                   linewidth=0, antialiased=False)
            ax.set_xlabel('x (um)')
            ax.set_ylabel('z (um)')
            ax.set_zlabel("Transverse ({})".format(unit))
            fig.colorbar(surf, shrink=0.5, aspect=5)
            plt.show()

        return x_grids, z_grids, xkicks, dE_dct

    def plot_wakes_contour(self, s):
        ind = find_nearest_ind(self.position_list, s)

        print("plot longitudinal wakes at nearest point s  = {} m, step count {}".format(self.position_list[ind],
                                                                                         self.step_list[ind]))

        step = "step_" + str(self.step_list[ind])

        with h5py.File(self.wake_filename, "r") as f:
            print("ebeam energy {}".format(f[step].attrs['beam_energy']))
            dE_dct = np.array(f[step]['longitudinal']['dE_dct'])
            unit = f[step]['longitudinal'].attrs['unit']

            x_grids = np.array(f[step]['longitudinal']['x_grids']).reshape(dE_dct.shape)
            z_grids = np.array(f[step]['longitudinal']['z_grids']).reshape(dE_dct.shape)

            xkicks = np.array(f[step]['transverse']['xkicks'])

        plot_2D_contour(x_grids,z_grids,dE_dct)
        plot_2D_contour(x_grids,z_grids,xkicks)

    def plot_particles(self, s, xkey='z', ykey='delta'):
        ind = find_nearest_ind(self.position_list, s)

        print("plot longitudinal wakes at nearest point s  = {} m, step count {}".format(self.position_list[ind],
                                                                                         self.step_list[ind]))

        step = "step_" + str(self.step_list[ind])

        with h5py.File(self.particle_filename, "r") as f:
            x = np.array(f[step]['particles'][xkey])
            y = np.array(f[step]['particles'][ykey])
            
        plt.figure()

        h, xedges, yedges, _ = plt.hist2d(x, y, bins=100, cmap=plt.cm.jet)
        plt.colorbar()
        plt.xlabel(xkey)
        plt.ylabel(ykey)
        plt.show()






