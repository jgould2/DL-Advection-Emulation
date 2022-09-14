r"""
One-dimensional advection with variable velocity
================================================

Solve the conservative variable-coefficient advection equation:

.. math:: q_t + (u(x)q)_x = 0.

Here q is the density of some conserved quantity and u(x) is the velocity.
The velocity field used is

.. math:: u(x) = 2 + sin(2\pi x).

The boundary conditions are periodic.
The initial data get stretched and compressed as they move through the
fast and slow parts of the velocity field.
"""


from __future__ import absolute_import
import numpy as np
from clawpack.pyclaw import plot
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow.keras.metrics as metrics
from DL_Models.Unet.Unet_base import model4
from clawpack import pyclaw
from clawpack import riemann
import threading
import os

os.environ['NUMEXPR_MAX_THREADS'] = "40"

def qinit_train(state):

    # Initial Data parameters

    x, y = state.p_centers

    #Two Guassian pulses for IC
    decay = np.random.normal(10, 5, (2, 2))
    center = np.random.uniform(.2, .8, (2, 2))
    amplitude = np.random.normal(1, .1, (2,))
    state.q[0, :, :] = amplitude[0] * np.exp(-decay[0, 0] * np.square(x - center[0, 0]) - decay[0, 1] * np.square(y - center[0, 1])) +\
                       amplitude[1] * np.exp(-decay[1, 0] * np.square(x - center[1, 0]) - decay[1, 1] * np.square(y - center[1, 1]))

def qinit_test(state):

    # Initial Data parameters

    x, y = state.p_centers

    #Two Guassian pulses for IC

    decay = [[8, 12], [12, 8]]
    center = [[.2, .2], [.8, .8]]
    amplitude = [.9, 1.1]

    state.q[0, :, :] = amplitude[0] * np.exp(-decay[0][0] * np.square(x - center[0][0]) - decay[0][1] * np.square(y - center[0][1])) +\
                       amplitude[1] * np.exp(-decay[1][0] * np.square(x - center[1][0]) - decay[1][1] * np.square(y - center[1][1]))

def auxinit_train(state):
    # Initilize petsc Structures for aux
    x, y = state.grid.p_centers
    freq = np.random.normal(2*np.pi, .2, (2, 2))
    offset = np.random.normal(0, .2, (2,))
    state.aux[0, :, :] = np.sin(freq[0, 0] * x)**2 + np.cos(freq[0, 1] * y) + offset[0]
    state.aux[1, :, :] = np.cos(freq[1, 0] * y)**2 + np.sin(freq[1, 1] * x) + offset[1]

def auxinit_test(state):
    # Initilize petsc Structures for aux
    x, y = state.grid.p_centers
    state.aux[0, :, :] = np.sin(np.pi*x)**2 * np.sin(2*np.pi*y)
    state.aux[1, :, :] = np.sin(np.pi*y)**2 * np.sin(2*np.pi*x)

def auxinit_const(state, constant=0):
    # Initilize petsc Structures for aux
    x, y = state.grid.p_centers
    state.aux[0, :, :] = np.sin(np.pi*x)**2 * np.sin(2*np.pi*y) * np.cos(np.pi*constant)
    state.aux[1, :, :] = np.sin(np.pi*y)**2 * np.sin(2*np.pi*x) * np.cos(np.pi*constant)


def setup(dimensions, domain, nsteps=256, tfinal_=1.0, training=False, constant=False, use_petsc=False,solver_type='sharpclaw',kernel_language='Fortran',outdir='/home/jacob/PycharmProjects/DL_Advection_Emulation/_output', weno_order=5):
    from clawpack import riemann

    if use_petsc:
        import clawpack.petclaw as pyclaw
    else:
        from clawpack import pyclaw

    if solver_type=='classic':
        solver = pyclaw.ClawSolver2D(riemann.vc_advection_2D)
    elif solver_type=='sharpclaw':
        solver = pyclaw.SharpClawSolver2D(riemann.vc_advection_2D)
        #solver.weno_order=weno_order
    else: raise Exception('Unrecognized value of solver_type.')

    solver.kernel_language = kernel_language
    solver.lim_type = 2
    solver.weno_order = 5
    solver.fwave = True
    # ‘SSP104’ : 4th-order strong stability preserving time-integration method - Ketcheson
    solver.time_integrator = 'SSP104'
    solver.bc_lower[0] = pyclaw.BC.periodic
    solver.bc_upper[0] = pyclaw.BC.periodic
    solver.bc_lower[1] = pyclaw.BC.periodic
    solver.bc_upper[1] = pyclaw.BC.periodic
    solver.aux_bc_lower[0] = pyclaw.BC.periodic
    solver.aux_bc_upper[0] = pyclaw.BC.periodic
    solver.aux_bc_lower[1] = pyclaw.BC.periodic
    solver.aux_bc_upper[1] = pyclaw.BC.periodic

    solver.cfl_max = 2.5
    #solver.cfl_desired = 1.0
    #solver.dt_initial=.00001

    xlower = domain[0][0]
    xupper = domain[0][1]
    mx = dimensions[0]
    ylower = domain[1][0]
    yupper = domain[1][1]
    my = dimensions[1]
    x = pyclaw.Dimension(xlower, xupper, mx, name='x')
    y = pyclaw.Dimension(ylower, yupper, my, name='y')
    domain = pyclaw.Domain([x, y])
    num_aux = 2
    num_eqn = 1
    state = pyclaw.State(domain, num_eqn, num_aux)

    if training:
        auxinit_train(state)
        qinit_train(state)
    elif constant != -1:
        auxinit_const(state, constant)
        qinit_const(state)
    else:
        auxinit_test(state)
        qinit_test(state)

    claw = pyclaw.Controller()
    claw.outdir = outdir
    claw.solution = pyclaw.Solution(state, domain)

    claw.solver = solver
    #claw.output_style = 3
    #claw.nstepout = 4
    claw.num_output_times = nsteps
    claw.tfinal = tfinal_
    #claw.output_style = 2
    #claw.out_times = [0.0, 1.0]
    if not constant:
        claw.write_aux_init = True
    #claw.setplot = setplot
    claw.keep_copy = True
    print('********', claw.solver.dt)
    claw.run()
    print("^^^^^^^^", claw.solver.dt)
    #claw.plot()
    return claw

#--------------------------
def setplot(plotdata):
#--------------------------
    """
    Specify what is to be plotted at each frame.
    Input:  plotdata, an instance of visclaw.data.ClawPlotData.
    Output: a modified version of plotdata.
    """
    plotdata.clearfigures()  # clear any old figures,axes,items data

    # Figure for q[0]
    plotfigure = plotdata.new_plotfigure(name='q', figno=1)

    # Set up for axes in this figure:
    plotaxes = plotfigure.new_plotaxes()
    plotaxes.ylimits = [-.1,1.1]
    plotaxes.title = 'q'

    # Set up for item on these axes:
    plotitem = plotaxes.new_plotitem(plot_type='1d_plot')
    plotitem.plot_var = 0
    plotitem.plotstyle = '-o'
    plotitem.color = 'b'
    plotitem.kwargs = {'linewidth':2,'markersize':5}

    return plotdata

def stats(data):
    print(np.mean(data), np.std(data), np.min(data), np.max(data))
    print("***************")
    return
def converge_resolution(dimensions, domain):
    # generate a random solution
    mape_list = [1]
    setup(dimensions, domain, tfinal_=1/256, nsteps=1, constant=0)
    prev_ = np.array(pd.read_fwf('/home/jacob/PycharmProjects/DL_Advection_Emulation/_output/fort.q0001')['1'].iloc[7:], dtype=np.float64).reshape(dimensions)
    dimensions_new = dimensions
    count=1
    print('(((((((((')
    #exit()
    while mape_list[-1] > 1e-6:
        dimensions_new = [2*i for i in dimensions_new]
        claw = setup(dimensions_new, domain, tfinal_=1 / 256, nsteps=1, constant=0).frames[-1].q[0]

        data = setup(dimensions_new, domain, tfinal_=1 / 256, nsteps=1, constant=0).frames[-1].aux
        print(data.shape, '%%%%%%%%%%%')
        #data = pd.read_fwf('/home/jacob/PycharmProjects/DL_Advection_Emulation/_output/fort.q0001')
        new_ = np.array(data['1'].iloc[7:], dtype=np.float64).reshape(dimensions_new)
        new_l = new_[2 ** (count - 1) - 1: dimensions_new[0] - 2 ** (count - 1): 2**count, 2 ** (count - 1) - 1: dimensions_new[1] - 2 ** (count - 1): 2**count]
        new_r = new_[2 ** (count - 1): dimensions_new[0] - 2 ** (count - 1) + 1: 2**count, 2 ** (count - 1): dimensions_new[1] - 2 ** (count - 1) + 1: 2**count]
        new_ = (new_l + new_r)/2
        mape_list.append(np.mean(np.abs(new_ - prev_) / (new_ + 1.0e-6)))
        prev_ = new_
        print(mape_list, count)
        count += 1
        break
    print(mape_list)
    return
def down_sample(solution, dimensions, factor):
    if factor > 1:
        new_l_l = solution[int(factor / 2) - 1: dimensions[0] - int(factor / 2): factor, int(factor / 2) - 1: dimensions[1] - int(factor / 2): int(factor)]
        new_l_r = solution[int(factor / 2) - 1: dimensions[0] - int(factor / 2): factor, int(factor / 2): dimensions[1] - int(factor / 2) + 1: int(factor)]
        new_r_l = solution[int(factor / 2): dimensions[0] - int(factor / 2) + 1: factor, int(factor / 2) - 1: dimensions[1] - int(factor / 2): int(factor)]
        new_r_r = solution[int(factor / 2): dimensions[0] - int(factor / 2) + 1: factor, int(factor / 2): dimensions[1] - int(factor / 2) + 1: int(factor)]
        return (new_l_l + new_l_r + new_r_l + new_r_r) / 4
    elif factor == 1:
        return solution
    else:
        print('down-sampling factor error OOB')
def gen_data(dimensions=(8, 8), domain_=[[0, 1], [0, 1]], nsteps=256, tfinal=1.0, mult=8, num_train=20000, num_val=5000, num_threads=1, outdir='/home/jacob/PycharmProjects/DL_Advection_Emulation/Claw_data'):
    # generate 'num_train' training pairs
    # run training for model by taking 1 timestep of size tfinal/nsteps at 8x resolution for a numerical model
    # - 8x resolution is the baseline solution
    ref_dimensions = [8 * i for i in dimensions]
    print('generating random training examples')
    solver = pyclaw.SharpClawSolver2D(riemann.vc_advection_2D)
    solver.kernel_language = 'Fortran'
    solver.lim_type = 2
    solver.weno_order = 5
    solver.fwave = True
    # ‘SSP104’ : 4th-order strong stability preserving time-integration method - Ketcheson
    solver.time_integrator = 'SSP104'
    solver.bc_lower[0] = pyclaw.BC.periodic
    solver.bc_upper[0] = pyclaw.BC.periodic
    solver.bc_lower[1] = pyclaw.BC.periodic
    solver.bc_upper[1] = pyclaw.BC.periodic
    solver.aux_bc_lower[0] = pyclaw.BC.periodic
    solver.aux_bc_upper[0] = pyclaw.BC.periodic
    solver.aux_bc_lower[1] = pyclaw.BC.periodic
    solver.aux_bc_upper[1] = pyclaw.BC.periodic

    solver.cfl_max = 2.5
    # solver.cfl_desired = 1.0
    # solver.dt_initial=.00001

    xlower = domain_[0][0]
    xupper = domain_[0][1]
    mx = ref_dimensions[0]
    ylower = domain_[1][0]
    yupper = domain_[1][1]
    my = ref_dimensions[1]
    x = pyclaw.Dimension(xlower, xupper, mx, name='x')
    y = pyclaw.Dimension(ylower, yupper, my, name='y')
    domain = pyclaw.Domain([x, y])
    num_aux = 2
    num_eqn = 1
    for i in range(7600, num_train):
        state = pyclaw.State(domain, num_eqn, num_aux)
        print(i)
        auxinit_train(state)
        qinit_train(state)
        claw = pyclaw.Controller()
        claw.outdir = outdir + '/_output'
        claw.solution = pyclaw.Solution(state, domain)
        claw.solver = solver
        claw.num_output_times = 1
        claw.tfinal = tfinal / nsteps
        claw.keep_copy = True
        claw.run()
        inp = np.expand_dims(down_sample(claw.frames[0].q[0], dimensions=ref_dimensions, factor=8), axis=-1)
        out = np.expand_dims(down_sample(claw.frames[-1].q[0], dimensions=ref_dimensions, factor=8), axis=-1)
        u = np.expand_dims(down_sample(claw.frames[0].aux[0], dimensions=ref_dimensions, factor=8), axis=-1)
        v = np.expand_dims(down_sample(claw.frames[0].aux[1], dimensions=ref_dimensions, factor=8), axis=-1)
        np.save(outdir + '/training/ex_' + str(i) + '.npy', np.concatenate([inp, u, v, out], axis=-1))


    print('Done generating training data, starting with validation data')

    for i in range(num_val):
        print(i)
        state = pyclaw.State(domain, num_eqn, num_aux)

        auxinit_train(state)
        qinit_train(state)

        claw = pyclaw.Controller()
        claw.outdir = outdir + '/_output'
        claw.solution = pyclaw.Solution(state, domain)
        claw.solver = solver
        claw.num_output_times = 1
        claw.tfinal = tfinal / nsteps
        claw.keep_copy = True
        claw.run()

        inp = np.expand_dims(down_sample(claw.frames[0].q[0], dimensions=ref_dimensions, factor=8), axis=-1)
        out = np.expand_dims(down_sample(claw.frames[-1].q[0], dimensions=ref_dimensions, factor=8), axis=-1)
        u = np.expand_dims(down_sample(claw.frames[0].aux[0], dimensions=ref_dimensions, factor=8), axis=-1)
        v = np.expand_dims(down_sample(claw.frames[0].aux[1], dimensions=ref_dimensions, factor=8), axis=-1)
        print(np.concatenate([inp, u, v, out], axis=-1))
        np.save(outdir+'/validation/ex_'+str(i)+'.npy', np.concatenate([inp, u, v, out], axis=-1))
    print('Done generating training and validation data, starting with the generation of simulations for test cases')
    # run evaluation for model at its original training resolution as well as at double and half the resolution
    for mult in [1/2, 1, 2, 4, 8, 16]:
        print(mult)
        # gen solution
        solver = pyclaw.SharpClawSolver2D(riemann.vc_advection_2D)
        solver.kernel_language = 'Fortran'
        solver.lim_type = 2
        solver.weno_order = 5
        solver.fwave = True
        # ‘SSP104’ : 4th-order strong stability preserving time-integration method - Ketcheson
        solver.time_integrator = 'SSP104'
        solver.bc_lower[0] = pyclaw.BC.periodic
        solver.bc_upper[0] = pyclaw.BC.periodic
        solver.bc_lower[1] = pyclaw.BC.periodic
        solver.bc_upper[1] = pyclaw.BC.periodic
        solver.aux_bc_lower[0] = pyclaw.BC.periodic
        solver.aux_bc_upper[0] = pyclaw.BC.periodic
        solver.aux_bc_lower[1] = pyclaw.BC.periodic
        solver.aux_bc_upper[1] = pyclaw.BC.periodic

        solver.cfl_max = 2.5
        # solver.cfl_desired = 1.0
        # solver.dt_initial=.00001

        xlower = domain_[0][0]
        xupper = domain_[0][1]
        mx = dimensions[0] * mult
        ylower = domain_[1][0]
        yupper = domain_[1][1]
        my = dimensions[1] * mult
        x = pyclaw.Dimension(xlower, xupper, mx, name='x')
        y = pyclaw.Dimension(ylower, yupper, my, name='y')
        domain = pyclaw.Domain([x, y])
        num_aux = 2
        num_eqn = 1
        state = pyclaw.State(domain, num_eqn, num_aux)

        auxinit_test(state)
        qinit_test(state)

        for i in range(nsteps):

            claw = pyclaw.Controller()
            claw.outdir = outdir + '/_output'
            claw.solution = pyclaw.Solution(state, domain)
            claw.solver = solver
            claw.num_output_times = 1
            claw.tfinal = tfinal / nsteps
            claw.keep_copy = True

            claw.run()

            data = claw.frames[-1].q[0]

            if mult >= 1/2 and mult <= 4:
                np.save(outdir+'/testing/'+str(int(mult * dimensions[0])) + '->' + str(int(dimensions[0] / 2)) + '_step:' + str(i) + '.npy', down_sample(data, [i*mult for i in dimensions], int(mult * 2)))
            if mult >= 1 and mult <= 8:
                np.save(outdir+'/testing/' + str(int(mult * dimensions[0])) + '->' + str(int(dimensions[0])) + '_step:' + str(i) + '.npy', down_sample(data, [i*mult for i in dimensions], int(mult)))
            if mult >= 2 and mult <= 16:
                np.save(outdir+'/testing/' + str(int(mult * dimensions[0])) + '->' + str(int(dimensions[0] * 2)) + '_step:' + str(i) + '.npy', down_sample(data, [i*mult for i in dimensions], int(mult / 2)))

            state = pyclaw.State(domain, num_eqn, num_aux)
            x, y = state.grid.p_centers
            state.aux[0, :, :] = np.sin(np.pi * x) ** 2 * np.sin(2 * np.pi * y) * np.cos(np.pi * claw.solver.dt * (i + 1) / claw.tfinal)
            state.aux[1, :, :] = np.sin(np.pi * y) ** 2 * np.sin(2 * np.pi * x) * np.cos(np.pi * claw.solver.dt * (i + 1) / claw.tfinal)
            state.q[0, :, :] = data

    return
def train_eval(model, dimensions, domain, nsteps, tfinal=1.0):
    # initialize model for the input shape we're training at
    nn_model = model(input_size=(dimensions[0], dimensions[1], 3))
    # generate random numerical solutions at 8x resolution - save to file for training
    dimensions_8x = [8*i for i in dimensions]

    print(model.summary())
if __name__=="__main__":
    #from clawpack.pyclaw.util import run_app_from_main
    #output = run_app_from_main(setup,setplot)
    #print(output)
    #train_eval(model=model4, dimensions=(64, 64), domain=[[0, 1], [0, 1]], nsteps=256, tfinal=1.0)
    exit()
    gen_data(dimensions=(64, 64))
    exit()
    converge_resolution([256, 256], [[0.0, 1.0], [0.0, 1.0]])
    exit()
    setup()
    data = pd.read_fwf('/home/jacob/PycharmProjects/DL_Advection_Emulation/_output/fort.q0000')
    aux = pd.read_fwf('/home/jacob/PycharmProjects/DL_Advection_Emulation/_output/fort.a0000')
    data1 = pd.read_fwf('/home/jacob/PycharmProjects/DL_Advection_Emulation/_output/fort.q0001')
    data2 = pd.read_fwf('/home/jacob/PycharmProjects/DL_Advection_Emulation/_output/fort.q0002')
    print(aux.head(20))
    u = np.array(aux['1'].iloc[7:], dtype=np.float64).reshape((232, 396))
    v = np.array(aux['patch_number'].iloc[7:], dtype=np.float64)
    inp = np.array(data['1'].iloc[7:], dtype=np.float64)
    out = np.array(data2['1'].iloc[7:], dtype=np.float64)
    print(u.shape, v.shape, inp.shape, out.shape)
    exit()
    stats(u)
    stats(v)
    stats(inp)
    stats(out)
    print(np.concatenate([inp, u, v, out], axis=-1).shape)
    exit()
    setup()
    data = pd.read_fwf('/home/jacob/PycharmProjects/DL_Advection_Emulation/_output/fort.q0000')
    data1 = pd.read_fwf('/home/jacob/PycharmProjects/DL_Advection_Emulation/_output/fort.q0001')
    data2 = pd.read_fwf('/home/jacob/PycharmProjects/DL_Advection_Emulation/_output/fort.q0002')
    stats(data['1'].iloc[7:])
    stats(data1['1'].iloc[7:])
    stats(data2['1'].iloc[7:])
    exit()
    for i in range(11000):
        filename = ''
        if i < 10000:
            filename = '/home/jacob/PycharmProjects/DL_Advection_Emulation/Claw_data/training/ex_' + str(i) + '.npy'
        else:
            filename = '/home/jacob/PycharmProjects/DL_Advection_Emulation/Claw_data/validation/ex_' + str(i - 10000) + '.npy'
        read_in = np.load(filename)
        data = np.concatenate([np.expand_dims(read_in[:, j * int(read_in.shape[-1] / 4):(j + 1) * int(read_in.shape[-1] / 4)], axis=-1) for j in range(4)], axis=-1)
        new1 = np.concatenate([np.expand_dims(down_sample(data[:, :, k], (512, 512), 8), axis=-1) for k in range(4)], axis=-1)
        print(new1.shape)
        np.save(filename, down_sample(data[:, :, 0], (512, 512), 8))
    #setup()