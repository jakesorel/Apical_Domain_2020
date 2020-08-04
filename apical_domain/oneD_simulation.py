import os
import numpy as np
from scipy.integrate import odeint
from matplotlib import cm
from joblib import Parallel, delayed
import multiprocessing
import time
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt
plt.rcParams.update({'pdf.fonttype': 42})


class simulation:
    """
    Simulator object for regular equations.

    Uses finite-difference method of solving PDE defined in Supplementary Modelling
    """
    def __init__(self):
        self.D_E = 0.100673828125
        self.k_off = 0.01755789881449827
        self.k_on = 1.88*10**-2
        self.E_crit = 6.89334310e+01
        self.PIP2_tot = 2.67800940e+03

        self.L = 100
        self.l_apical = 37.0492

        self.num_x = []
        self.num_apical = 100
        self.dx = []

        self.dt = []
        self.tfin = []
        self.t_span = []

        self.y0 = []

        self.ysol = []
        self.ysolorig = []
        self.ysolshift = []
        self.ysol_apical = []

        self.make_directory("plots")

        self.pol_thresh = 40

        self.apical_mask = []

        self.rel_height = []
        self.rel_heights = []
        self.amount = []
        self.peaks = []




    def make_directory(self,dir):
        """
        Makes a new directory, specified by the string dir

        :param dir: Directory name (**string**)
        """
        if not os.path.exists(dir):
            os.mkdir(dir)

    def set_num_x(self,n,apical_on=False):
        """
        Sets spatial discretisation of number **num_x**

        if **apical_on** is **True**, then **n** specifies the number of spatial blocks within the apical domain
        (of length **self.l_apical**), and **self.num_x** is computed with respect to the ratio of **self.l_apical** and
        **self.L** (the full domain length i.e. cell perimeter)

        if **apical_on** is **False**, then **n** defines **self.num_x**

        :param n: Number of spatial blocks (**np.int32**)
        :param apical_on: Specifies whether apical membrane is considered specifically or not (**np.bool**)
        :return:
        """
        if apical_on is True:
            self.num_apical = n
            self.num_x = int(self.L / self.l_apical * self.num_apical)
            self.apical_mask = np.zeros(self.num_x)
            self.apical_mask[int(self.num_x/2 - self.num_apical/2):int(self.num_x/2 + self.num_apical/2)] = 1
        else:
            self.num_x = n
        self.dx = self.L / self.num_x

    def set_t_span(self,dt,tfin):
        """
        Sets temporal discretisation. Saves a 1D array of time-points to **self.t_span**

        :param dt: Time-step (**np.float32**)
        :param tfin: Final time-step (**np.float32**)
        :return:
        """
        self.dt, self.tfin = dt, tfin
        self.t_span = np.arange(0, tfin + dt, dt)


    def set_initial(self,mean = False,SD = False,override=False,apical_on = False):
        """
        Set the initial conditions of **E**.

        If **override** is **False**, then the initial condition of **E**, **self.y0**, is given by a normal distribution,
        with mean **mean** and standard deviation **SD**

        If **overide** is not **False** but instead a 1D array of length (**self.num_x**), then **self.y0** is defined as
        **override**

        If **apical_on** is **True**, then outside the apical membrane, **self.y0** is set to 1e-17 (<<1)

        :param mean: Mean value used in normal distribution (**np.float32**)
        :param SD: Standard deviation used in normal distribution (**np.float32**)
        :param override: Override 1D array if provided (**np.ndarray** of dtype **np.float32**), else **False**
        :param apical_on: Sets **self.y0** to (approx.) 0 outside the apical membrane.
        :return:
        """
        if override is False:
            self.y0 = np.random.normal(mean,SD,self.num_x)
        else:
            self.y0 = override
        if apical_on is True:
            self.y0 = self.y0*self.apical_mask + 1e-17

    def del_1d(self,y, dx):
        """
        Central difference discretisation of the Laplacian.

        :param y: Spatially discretised 1D function (i.e. a **np.ndarray** of dtype **np.float32**) on which the function is applied
        :param dx: Spatial discretisation lengthscale (**np.float32**)
        :return: Spatially discretised Laplacian of y
        """
        return (np.roll(y, -1) - 2*y + np.roll(y, 1)) / dx**2

    def f(self,y, t):
        """
        System of partial differential equations as defined in the Supplementary Modelling, describing the spatio-temporal
        evolution of **E**.

        :param y: Spatially discretised 1D function of **E** (i.e. a **np.ndarray** of dtype **np.float32**) of length **self.num_x**
        :param t: Time-point at which to evaluate (**np.float32**)
        :return: Time differential of **E** at each spatial block (**np.ndarray** of dtype **np.float32** and length **self.num_x**)
        """
        E = y
        PIP2_free = self.PIP2_tot - np.sum(E) * self.dx
        dE = self.D_E * self.del_1d(E, self.dx) + self.k_on * PIP2_free / (1 + (self.E_crit / E) ** 2) - self.k_off * E
        return dE

    def f_apical_on(self,y, t):
        """
        System of partial differential equations as defined in the Supplementary Modelling, describing the spatio-temporal
        evolution of **E** **modelling cell-cell contacts, where E can load only on the apical membrane**.

        :param y: Spatially discretised 1D function of **E** (i.e. a **np.ndarray** of dtype **np.float32**) of length **self.num_x**
        :param t: Time-point at which to evaluate (**np.float32**)
        :return: Time differential of **E** at each spatial block (**np.ndarray** of dtype **np.float32** and length **self.num_x**)
        """
        E = y
        PIP2_free = self.PIP2_tot - np.sum(E) * self.dx
        dE = self.D_E * self.del_1d(E, self.dx) + self.apical_mask*self.k_on * PIP2_free / (1 + (self.E_crit / E) ** 2) - self.k_off * E
        return dE

    def solve(self,apical_on=False):
        """
        Perform the simulation. Uses the **scipy.integrate.odeint** package to integrate the defined system of PDEs.

        If **apical_on** is **True**, then simulate with cell-cell contacts (where loading is restricted to the apical membrane)

        If **apical_on** is **False**, then simulate without cell-cell contacts.

        :param apical_on: Specifies whether to simulate with or without cell-cell contacts (**np.bool**)
        :return: **self.y_sol**, a 2D **np.ndarray** array (**n_t** x **self.num_x**), where **n_t** is the number of time-steps (as defined by **self.t_span**).
        """
        if apical_on is False:
            self.ysol = odeint(self.f, self.y0, self.t_span)
        else:
            self.ysol = odeint(self.f_apical_on, self.y0, self.t_span)
        self.ysolorig = self.ysol
        return self.ysol

    def centre_solution(self):
        """
        Given the simulation is performed on Periodic Boundary Conditions, the central location is undefined. For clarity,
        this function centres the solution such that the maximal **E** at the final time-step lies at **self.num_x/2**.

        Over-writes **self.y_sol** and also saves in **self.ysolshift** as a copy.
        """
        x_max = np.mean(np.where(self.ysol[-1] == self.ysol[-1].max())[0])
        x_shift = int(self.num_x / 2 - x_max)
        self.ysolshift = np.roll(self.ysol, x_shift, axis=1)
        self.ysol = self.ysolshift

    def get_apical_solution(self):
        """
        Crop the solution down to just the apical membrane (as defined by **self.l_apical** versus **self.L**).

        The solution is first centred before cropping.
        """
        self.centre_solution()
        self.ysol_apical = self.ysolshift[:, int(self.num_x / 2 - self.num_apical / 2):int(self.num_x / 2 + self.num_apical / 2)]

    def norm(self,x):
        """
        Generic function to normalize a 1D **np.ndarray**, calculating the normalized array **y** as:

        y = (x-min(x))/(max(x) - min(x))

        :param x: A 1D **np.ndarray** to be normalized.
        :return: Normalized **np.ndarray**, **y**.
        """
        return (x-x.min())/(x.max()-x.min())

    def plot_time_series(self,cmap,show=True,filename=False,apical=False,ylim=(0,150)):
        """
        Plot time-series of the simulation, as an overlayed set of lines.

        The number of time-points that are sampled is set by the parameter **cmap**, a (n_sample x 4) **np.ndarray** of
        RGBA colour points. **cmap** can be generated using plt.cm.Reds(np.linspace(0,1,n_sample)) for example.

        :param cmap: The colormap used to plot the solution (a **np.ndarray**)
        :param show: If **True**, then show the plot.
        :param filename: **str** defining the file-name of the plot if is being saved. If **False** then the plot is not saved.
        :param apical: Determines whether to plot the whole membrane (**False**) or just the apical membrane (**True**) (**np.bool**)
        :param ylim: Axis limits on the y-axis (i.e. the concentration of **E**) (**tuple**)
        """
        self.centre_solution()
        if apical is True:
            self.get_apical_solution()

        N = cmap.shape[0]

        fig, ax = plt.subplots(figsize=(3.8, 3))
        my_cmap = ListedColormap(cmap)

        for i in range(N):
            t = int(self.t_span.size * i / N)
            if apical is False:
                ax.plot(np.linspace(0, self.L, self.num_x), self.ysol[t], color=cmap[i])
            else:
                ax.plot(np.linspace(0, self.l_apical, self.num_apical), self.ysol_apical[t], color=cmap[i])
        ax.set(xlabel=r"x", ylabel=r"$E$",ylim=ylim)
        sm = plt.cm.ScalarMappable(cmap=my_cmap, norm=plt.Normalize(vmax=100, vmin=0))
        sm._A = []
        cl = plt.colorbar(sm, ax=ax, pad=0.05, fraction=0.1, aspect=10, orientation="vertical")
        cl.set_label("Time (%)")
        fig.subplots_adjust(top=0.8, bottom=0.2, left=0.2, right=0.8)
        if show is True:
            fig.show()
        if filename is not False:
            fig.savefig("plots/%s.pdf"%filename,dpi=300)

    def get_rel_height(self):
        """
        Find ∆E, the difference between the maximum and minimum value of **E** at the final time-step.

        :return: ∆E, **self.rel_height** (a **np.float32**)
        """
        self.rel_height = self.ysol[-1].max() - self.ysol[-1].min()
        return self.rel_height

    def get_rel_heights(self):
        """
        Find ∆E for all time-points

        :return: **self.rel_heights (**np.ndarray** of shape (**n_t** x 1) and dtype **np.float32**)
        """
        self.rel_heights = np.max(self.ysol, axis=1) - np.min(self.ysol, axis=1)
        return self.rel_heights

    def get_amount(self):
        """
        Find **self.amount** = int_0^L {E} dx

        :return: **self.amount**
        """
        self.amount = np.sum(self.ysol[-1]) * self.dx
        return self.amount

    def find_peaks(self, y):
        """
        Count the number of peaks in a solution.

        :param y: **1D** array of concentrations (i.e. **E** at a given time t) (**np.ndarray** of size (**self.num_x x 1**) and dtype **np.float32**)
        :return: Number of peaks (**np.int32**)
        """
        ynorm = y - y.min()
        ysign = np.sign(ynorm - (y.max()) / 2)
        pks = int(np.sum((ysign + np.roll(ysign, 1)) == 0) / 2)
        return pks

    def get_peaks(self):
        """
        Find the number of peaks for all time-points.

        Number of peaks = 0 when the relative height is too low (i.e. discounting spurious multi-peak solutions, where variations are miniscule).
        The threshold is set by **self.pol_thresh**.

        :return: Number of peaks **self.peaks** a **np.ndarray** of size (**n_t** x 1) and dtype **np.int32**
        """
        if type(self.rel_height) is list:
            self.get_rel_height()
        if self.rel_height > self.pol_thresh:
            self.peaks = (self.find_peaks(self.ysol[-1]) > 1) * 1.0
        else:
            self.peaks = 0.0
        return self.peaks


class phase_space:
    """
    Phase space object performing a parameter span for a (custom) pair of parameters using regular equations
    """
    def __init__(self):
        self.sim = simulation()
        self.sim.set_num_x(30, apical_on=False)
        self.sim.set_t_span(0.1, 100)
        self.recalculate_initial = True
        self.rep = 1

        self.xvar, self.yvar = [], []
        self.xname, self.yname = r"$log_10 \lambda$", r"$1 / \epsilon$"
        self.x_space, self.y_space = [], []
        self.rel_height, self.amount,self.peaks = [],[],[]

    def set_names(self, xname, yname):
        """
        Phase space considers solutions when varying two parameters. So this function sets the **axis-labels** of the
        x and y axes

        :param xname: **Axis label** of the x-axis (**str**), e.g. r"$log_10 \lambda$"
        :param yname: **Axis label** of the x-axis (**str**), e.g. r"$1 / \epsilon$"
        """
        self.xname, self.yname = xname, yname

    def get_initial(self):
        """
        Calculate initial condition for the simulation.

        If the parameter **self.recalculate_initial** is **True**, then the initial condition is recalculated, as defined
        in the **simulation** class.

        If it's **False**, then the previous initial condition is utilized**

        :return:
        """
        if self.recalculate_initial is True:
            self.sim.set_initial(40, 0.5)

    def get_outname(self, out):
        """
        Defines the labels attributed to the various statistics that are calculated.

        :param out: The output matrix from the parameter sweep (e.g. **self.rel_height**)
        :return: Label (**str**) (e.g. r"$\Delta E$")
        """
        if out is self.rel_height:
            return r"$\Delta E$"
        if out is self.amount:
            return r"$\int_0^L  E \ dx$"

    def get_stats(self, X):
        """
        Perform the simulation (with repeats as defined by the **np.int** **self.rep**), and calculate the summary statistics:
        the relative height (∆E), the amount (int E dx), and the number of peaks.

        :param X: Two-element list or array defining the values of the parameters attributed to the x or y axis
        :return: Three-element array of the statistics, as defined by the order listed above.
        """
        setattr(self.sim, self.xvar, X[0])
        setattr(self.sim, self.yvar, X[1])
        h, a, p = [],[],[]
        for i in range(self.rep):
            self.get_initial()
            self.sim.solve()
            h.append(self.sim.get_rel_height())
            a.append(self.sim.get_amount())
            p.append(self.sim.get_peaks())
        return np.array([np.nanmean(h), np.nanmean(a), np.nanmean(p)])

    def perform_param_scan(self, xscale, yscale,xlim, ylim,  resolution, xvar="D_E", yvar="E_crit",
                           recalculate_initial=True):
        """
        Perform the parameter scan.

        Saves the parameter-scan to an **npy** file.

        Uses the **joblib** parallelisation package.

        :param xlim: Tuple, setting the minimum and maximum of the xvar to simulate over.
            If xlim = (a,b), (where a < b)
            then if xscale is "log", simulate between 10^a and 10^b
            if xscale is "reciprocal", simulate between 1/b and 1/a
            if xscale is "normal", then simulate between a and b
        :param ylim: Tuple, setting the minimum and maximum of the yvar to simulate over. Same rules as xvar
        :param xscale: Scale of the x-axis, setting the stepping of the parameter discretization. A **str**: either "log", "reciprocal" or "normal"
        :param yscale: Scale of y-axis. Same rules as xscale.
        :param resolution: Number of parameter discretisations to simulate over. Can be an **np.int32**, where paramter
            discretisation number is same in x and y axis, or a **tuple** of **np.int32**, defining the number of
            spatial discretisations for the x vs y axis
        :param xvar: **str** defining the x-axis attributed parameter. Must match nomencalture of the code precisely. e.g. "D_E"
        :param yvar: **str** defining the y-axis attributed parameter. Must match nomencalture of the code precisely. e.g. "E_crit"
        :param recalculate_initial: Recalculate the initial condition of the simulation for each repeat and each
            parameter combination if **True**. Else the first initial condition is stored and re-used.

        """
        self.xvar, self.yvar = xvar, yvar
        self.sim.set_initial(40, 1)
        self.recalculate_initial = recalculate_initial
        if type(resolution) is tuple:
            resolutionx, resolutiony = resolution
        else:
            resolutionx,resolutiony = resolution, resolution
        if xscale == "log":
            self.x_space = 10.0 ** np.linspace(xlim[0], xlim[1], resolutionx)

        elif xscale == "reciprocal":
            self.x_space = 1 / np.flip(np.linspace(xlim[0], xlim[1], resolutionx))

        else:
            self.x_space = np.linspace(xlim[0], xlim[1], resolutionx)

        if yscale == "log":
            self.y_space = 10.0 ** np.linspace(ylim[0], ylim[1], resolutiony)

        elif yscale == "reciprocal":
            self.y_space = 1 / np.flip(np.linspace(ylim[0], ylim[1], resolutiony))

        else:
            self.y_space = np.linspace(ylim[0], ylim[1], resolutiony)

        X, Y = np.meshgrid(self.x_space, self.y_space, indexing="ij")

        inputs = np.array([X.ravel(), Y.ravel()]).T
        num_cores = multiprocessing.cpu_count()
        results = Parallel(n_jobs=num_cores)(delayed(self.get_stats)(X) for X in inputs)
        out = np.array(results).reshape(self.x_space.size, self.y_space.size, 3)
        self.rel_height, self.amount,  self.peaks = out.transpose(2, 0, 1)
        np.save("param_scan_%s.npy" % time.time(), out)

    def make_param_span_plot(self, ax, out, xscale="log", yscale="log", percentile_lim=(0, 100),
                             cmap=plt.cm.plasma,overlay_peaks=False):
        """
        Plot the parameter scan.

        :param ax: **matplotlib** axis object to plot the parameter scan onto.
        :param out: **np.ndarray** of statistics for each parameter combination (**resolution x resolution**). E.g. **self.rel_height**
        :param xscale: Scale of the x-axis. Must be the same as in **self.perform_param_scan**.
        :param yscale: Scale of the y-axis. Must be the same as in **self.perform_param_scan**.
        :param percentile_lim: Cut-off for the colorbar, setting the maximum and minimum percentile of the statistic for which a color is attributed.
        :param cmap: **matplotlib.cm** object e.g. **matplotlib.cm.plasma**
        :param overlay_peaks: Overlay the number of peaks for each parameter combination (i.e. to identify multi-peak regimes).
        """
        if xscale == "log":
            xlim = np.log10(self.x_space[0]), np.log10(self.x_space[-1])

        elif xscale == "reciprocal":
            xlim = 1 / self.x_space[0], 1 / self.x_space[-1]

        else:
            xlim = self.x_space[0], self.x_space[-1]

        if yscale == "log":
            ylim = np.log10(self.y_space[0]), np.log10(self.y_space[-1])

        elif yscale == "reciprocal":
            ylim = 1 / self.y_space[-1], 1 / self.y_space[0]

        else:
            ylim = self.y_space[0], self.y_space[-1]

        self.extent = [xlim[0], xlim[1], ylim[0], ylim[1]]
        vmin, vmax = np.nanpercentile(out, percentile_lim[0]), np.nanpercentile(out, percentile_lim[1])
        ax.imshow(np.flip(np.ones_like(out), axis=0), cmap=plt.cm.Greys, interpolation='nearest',
                  extent=self.extent, vmax=4,
                  vmin=0, zorder=-20)
        ax.imshow(np.flip(out.T, axis=0), cmap=cmap, interpolation='nearest', extent=self.extent, vmax=vmax,
                  vmin=vmin)
        if overlay_peaks is True:
            self.overlay_peaks(ax)
        sm = plt.cm.ScalarMappable(cmap=plt.cm.plasma, norm=plt.Normalize(vmax=vmax, vmin=vmin))
        sm._A = []
        cl = plt.colorbar(sm, ax=ax, pad=0.05, fraction=0.073, aspect=12, orientation="vertical")
        cl.set_label(self.get_outname(out))

        ax.set(xlabel=self.xname, ylabel=self.yname,
               aspect=(self.extent[0] - self.extent[1]) / (self.extent[2] - self.extent[3]))
        return ax

    def overlay_peaks(self, ax):
        """
        Overlay the number of peaks. Transparent if 1 or 0 peaks, or increasingly black if multi-peak solutions are frequent.

        :param ax: **matplotlib** axis object
        :return: ax
        """
        mask = np.flip(self.peaks.T, axis=0)
        mask = np.ma.masked_where((np.flip(self.peaks.T, axis=0) == 0.0), mask)
        ax.imshow(mask, cmap=cm.Greys, extent=self.extent, vmin=0, vmax=1)
        return ax

    def plot_param_scan(self, out, xscale="log", yscale="log", percentile_lim=(0, 100), cmap=plt.cm.plasma,save=True,overlay_peaks=False):
        """
        Make full parameter scan plot. A wrapper for **self.make_param_span_plot**.

        Saves the plot to the **plots** directory as a PDF.

        :param out: **np.ndarray** of statistics for each parameter combination (**resolution x resolution**). E.g. **self.rel_height**
        :param xscale: Scale of the x-axis. Must be the same as in **self.perform_param_scan**.
        :param yscale: Scale of the y-axis. Must be the same as in **self.perform_param_scan**.
        :param percentile_lim: Cut-off for the colorbar, setting the maximum and minimum percentile of the statistic for which a color is attributed.
        :param cmap: **matplotlib.cm** object e.g. **matplotlib.cm.plasma**
        :param overlay_peaks: Overlay the number of peaks for each parameter combination (i.e. to identify multi-peak regimes).

        """
        fig, ax = plt.subplots(figsize=(3, 3))
        self.make_param_span_plot(ax, out, xscale=xscale, yscale=yscale, percentile_lim=percentile_lim, cmap=cmap,overlay_peaks=overlay_peaks)
        fig.show()
        if save is True:
            fig.savefig("plots/%s vs %s (%s) %s.pdf"%(self.xname,self.yname,self.get_outname(out),overlay_peaks),dpi=300)