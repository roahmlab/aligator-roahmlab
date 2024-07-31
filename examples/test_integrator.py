# Below is modified from examples/ur5_reach.py

import aligator
import numpy as np
import time

import pinocchio as pin
import example_robot_data

from aligator import constraints, manifolds, dynamics  # noqa
from pinocchio.visualize import MeshcatVisualizer

from utils import ArgsBase, integrator_helper

import scipy.io


class Args(ArgsBase):
    plot: bool = True
    fddp: bool = False
    bounds: bool = False
    control: bool = True


args = Args().parse_args()

robot = example_robot_data.load("ur5")
rmodel: pin.Model = robot.model
rdata: pin.Data = robot.data
space = manifolds.MultibodyPhaseSpace(rmodel)

vizer = MeshcatVisualizer(rmodel, robot.collision_model, robot.visual_model, data=rdata)
vizer.initViewer(open=args.display, loadModel=True)
vizer.setBackgroundColor()


x0 = space.neutral()

ndx = space.ndx
nq = rmodel.nq
nv = rmodel.nv
nu = nv
q0 = x0[:nq]

if args.display:
    vizer.display(q0)

B_mat = np.eye(nu)

dt = 0.01
Tf = 100 * dt
nsteps = int(Tf / dt)

ode = dynamics.MultibodyFreeFwdDynamics(space, B_mat)
discrete_dynamics = dynamics.IntegratorSemiImplEuler(ode, dt)

wt_x = 1e-4 * np.ones(ndx)
wt_x[nv:] = 1e-2
wt_x = np.diag(wt_x)
wt_u = 1e-4 * np.eye(nu)


tool_name = "tool0"
tool_id = rmodel.getFrameId(tool_name)
target_pos = np.array([0.15, 0.65, 0.5])
print(target_pos)

frame_fn = aligator.FrameTranslationResidual(ndx, nu, rmodel, target_pos, tool_id)
v_ref = pin.Motion()
v_ref.np[:] = 0.0
frame_vel_fn = aligator.FrameVelocityResidual(
    ndx, nu, rmodel, v_ref, tool_id, pin.LOCAL
)
wt_x_term = wt_x.copy()
wt_x_term[:] = 1e-4
wt_frame_pos = 10.0 * np.eye(frame_fn.nr)
wt_frame_vel = 100.0 * np.ones(frame_vel_fn.nr)
wt_frame_vel = np.diag(wt_frame_vel)

term_cost = aligator.CostStack(space, nu)
term_cost.addCost(aligator.QuadraticCost(wt_x_term, wt_u * 0))
term_cost.addCost(aligator.QuadraticResidualCost(space, frame_fn, wt_frame_pos))
term_cost.addCost(aligator.QuadraticResidualCost(space, frame_vel_fn, wt_frame_vel))

u_max = rmodel.effortLimit
u_min = -u_max


def make_control_bounds():
    fun = aligator.ControlErrorResidual(ndx, nu)
    cstr_set = constraints.BoxConstraint(u_min, u_max)
    return aligator.StageConstraint(fun, cstr_set)


def computeQuasistatic(model: pin.Model, x0, a):
    data = model.createData()
    q0 = x0[:nq]
    v0 = x0[nq : nq + nv]

    return pin.rnea(model, data, q0, v0, a)


init_us = [computeQuasistatic(rmodel, x0, a=np.zeros(nv)) for _ in range(nsteps)]
init_xs = aligator.rollout(discrete_dynamics, x0, init_us)


stages = []

for i in range(nsteps):
    rcost = aligator.CostStack(space, nu)
    rcost.addCost(aligator.QuadraticCost(wt_x * dt, wt_u * dt))

    stm = aligator.StageModel(rcost, discrete_dynamics)
    if args.bounds:
        stm.addConstraint(make_control_bounds())
    stages.append(stm)


problem = aligator.TrajOptProblem(x0, stages, term_cost=term_cost)
tol = 1e-7

mu_init = 1e-7
rho_init = 0.0
verbose = aligator.VerboseLevel.VERBOSE
max_iters = 40
solver = aligator.SolverProxDDP(
    tol, mu_init, rho_init, max_iters=max_iters, verbose=verbose
)
solver.rollout_type = aligator.ROLLOUT_NONLINEAR
solver.sa_strategy = aligator.SA_LINESEARCH
if args.fddp:
    solver = aligator.SolverFDDP(tol, verbose, max_iters=max_iters)
cb = aligator.HistoryCallback()
solver.registerCallback("his", cb)
solver.setup(problem)
solver.run(problem, init_xs, init_us)


results = solver.results
print(results)

xs_opt = np.asarray(results.xs.tolist())
us_opt = np.asarray(results.us.tolist())


## forward simulation with integrator_helper
dt_sim = 1e-3

if args.control:
    # evaluation of the optimal control solution from alligator
    ts_opt = np.arange(0, Tf, dt)
    ts_sim = np.arange(0, ts_opt[-1], dt_sim)
    Kp = np.diag(30.0 * np.ones(nq))
    Kd = 0.05 * np.sqrt(Kp)
    sol, us_sim, e_sim, edot_sim = integrator_helper.integrate(rmodel, ts_sim, x0,
                                                               ts_opt, xs_opt, us_opt, B_mat,
                                                               Kp, Kd)
    
    scipy.io.savemat('data/ur5_reach.mat', {'ts_sim': ts_sim, \
                                            'sol': sol, \
                                            'us_sim': us_sim, \
                                            'e_sim': e_sim, \
                                            'edot_sim': edot_sim})
else:
    # naive test without any control
    ts_sim = np.arange(0, 10.0, dt_sim)
    sol = integrator_helper.integrate(rmodel, ts_sim, x0)

qs = [x[:nq] for x in sol]

if args.display:
    for _ in range(5):
        vizer.play(qs, dt_sim)
        time.sleep(0.5)
