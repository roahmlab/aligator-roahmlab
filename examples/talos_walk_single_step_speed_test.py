import numpy as np
import aligator
import pinocchio as pin
import time
import scipy.io

from aligator import (
    manifolds,
    dynamics,
    constraints,
)
from utils import load_talos_only_legs, ArgsBase, integrator_floatingbase_helper


class Args(ArgsBase):
    tcp: str = None
    bounds: bool = True
    num_threads: int = 8
    max_iters: int = 200
    step_length: float = 0.0
    dt: float = 0.01

args = Args().parse_args()
robotComplete, robot = load_talos_only_legs()
rmodel: pin.Model = robot.model
rdata: pin.Data = robot.data
nq = rmodel.nq
nv = rmodel.nv
nu = nv - 6
print("nq:", nq)
print("nv:", nv)

if args.display:
    vizer = pin.visualize.MeshcatVisualizer(
        rmodel, robot.collision_model, robot.visual_model, data=rdata
    )
    vizer.initViewer(open=True, loadModel=True)
    vizer.display(pin.neutral(rmodel))
    vizer.setBackgroundColor()

FOOT_FRAME_IDS = {
    fname: rmodel.getFrameId(fname) for fname in ["left_sole_link", "right_sole_link"]
}
FOOT_JOINT_IDS = {
    fname: rmodel.frames[fid].parentJoint for fname, fid in FOOT_FRAME_IDS.items()
}

controlled_joints = rmodel.names[1:].tolist()
controlled_ids = [
    robotComplete.model.getJointId(name_joint) for name_joint in controlled_joints[1:]
]
q0 = rmodel.referenceConfigurations["half_sitting"]
v0 = np.zeros(nv)

pin.forwardKinematics(rmodel, rdata, q0)
pin.updateFramePlacements(rmodel, rdata)

space = manifolds.MultibodyPhaseSpace(rmodel)

x0 = np.concatenate((q0, v0))
u0 = np.zeros(nu)
com0 = pin.centerOfMass(rmodel, rdata, x0[:nq])
dt = args.dt

# Define OCP weights
w_x = np.array(
    [
        0,
        0,
        0,
        10000,
        10000,
        10000,  # Base pos/ori
        1,
        1,
        1,
        1,
        1,
        1,  # Left leg
        1,
        1,
        1,
        1,
        1,
        1,  # Right leg
        100,
        100,
        100,
        100,
        100,
        100,  # Base pos/ori vel
        10,
        10,
        10,
        10,
        10,
        10,  # Left leg vel
        10,
        10,
        10,
        10,
        10,
        10   # Right leg vel
    ]
)
w_x = np.diag(w_x)
w_u = np.eye(nu) * 1e-3
w_LFRF = 100000 * np.eye(6)
w_com = 10000 * np.ones(3)
w_com = np.diag(w_com)

act_matrix = np.eye(nv, nu, -6)

# Create dynamics and costs
prox_settings = pin.ProximalSettings(1e-9, 1e-10, 10)
constraint_models = []
constraint_datas = []
for fname, fid in FOOT_FRAME_IDS.items():
    joint_id = FOOT_JOINT_IDS[fname]
    pl1 = rmodel.frames[fid].placement
    pl2 = rdata.oMf[fid]
    cm = pin.RigidConstraintModel(
        pin.ContactType.CONTACT_6D,
        rmodel,
        joint_id,
        pl1,
        0,
        pl2,
        pin.LOCAL_WORLD_ALIGNED,
    )
    cm.corrector.Kp[:] = (0, 0, 100, 0, 0, 0)
    cm.corrector.Kd[:] = (50, 50, 50, 50, 50, 50)
    constraint_models.append(cm)
    constraint_datas.append(cm.createData())


def create_dynamics(support):
    dyn_model = None
    if support == "LEFT":
        ode = dynamics.MultibodyConstraintFwdDynamics(
            space, act_matrix, [constraint_models[0]], prox_settings
        )
        dyn_model = dynamics.IntegratorSemiImplEuler(ode, dt)
    elif support == "RIGHT":
        ode = dynamics.MultibodyConstraintFwdDynamics(
            space, act_matrix, [constraint_models[1]], prox_settings
        )
        dyn_model = dynamics.IntegratorSemiImplEuler(ode, dt)
    else:
        ode = dynamics.MultibodyConstraintFwdDynamics(
            space, act_matrix, constraint_models, prox_settings
        )
        dyn_model = dynamics.IntegratorSemiImplEuler(ode, dt)
    return dyn_model


LF_id = rmodel.getFrameId("left_sole_link")
RF_id = rmodel.getFrameId("right_sole_link")
root_id = rmodel.getFrameId("root_joint")
LF_placement = rdata.oMf[LF_id]
RF_placement = rdata.oMf[RF_id]

frame_com = aligator.CenterOfMassTranslationResidual(space.ndx, nu, rmodel, com0)
v_ref = pin.Motion()
v_ref.np[:] = 0.0
frame_vel_LF = aligator.FrameVelocityResidual(
    space.ndx, nu, rmodel, v_ref, LF_id, pin.LOCAL
)
frame_vel_RF = aligator.FrameVelocityResidual(
    space.ndx, nu, rmodel, v_ref, RF_id, pin.LOCAL
)


def createStage(support, prev_support, LF_target, RF_target):
    frame_fn_LF = aligator.FramePlacementResidual(
        space.ndx, nu, rmodel, LF_target, LF_id
    )
    frame_fn_RF = aligator.FramePlacementResidual(
        space.ndx, nu, rmodel, RF_target, RF_id
    )
    frame_cs_RF = aligator.FrameTranslationResidual(
        space.ndx, nu, rmodel, RF_target.translation, RF_id
    )[2]
    frame_cs_LF = aligator.FrameTranslationResidual(
        space.ndx, nu, rmodel, LF_target.translation, LF_id
    )[2]

    rcost = aligator.CostStack(space, nu)
    rcost.addCost(aligator.QuadraticStateCost(space, nu, x0, w_x))
    rcost.addCost(aligator.QuadraticControlCost(space, u0, w_u))
    """ rcost.addCost(aligator.QuadraticResidualCost(space, frame_com, w_com)) """
    if support == "LEFT":
        rcost.addCost(aligator.QuadraticResidualCost(space, frame_fn_RF, w_LFRF))
    elif support == "RIGHT":
        rcost.addCost(aligator.QuadraticResidualCost(space, frame_fn_LF, w_LFRF))

    stm = aligator.StageModel(rcost, create_dynamics(support))
    umax = act_matrix.T @ rmodel.effortLimit
    umin = -umax
    if args.bounds:
        # print("Control bounds activated")
        # fun: u -> u
        ctrl_fn = aligator.ControlErrorResidual(space.ndx, np.zeros(nu))
        stm.addConstraint(ctrl_fn, constraints.BoxConstraint(umin, umax))

    if support == "DOUBLE" and prev_support == "LEFT":
        stm.addConstraint(frame_vel_RF, constraints.EqualityConstraintSet())
        stm.addConstraint(frame_cs_RF, constraints.EqualityConstraintSet())
    elif support == "DOUBLE" and prev_support == "RIGHT":
        stm.addConstraint(frame_vel_LF, constraints.EqualityConstraintSet())
        stm.addConstraint(frame_cs_LF, constraints.EqualityConstraintSet())

    return stm


term_cost = aligator.CostStack(space, nu)
term_cost.addCost(aligator.QuadraticStateCost(space, nu, x0, 100 * w_x))
""" term_cost.addCost(aligator.QuadraticResidualCost(space, frame_com, 100 * w_com)) """

# Define contact phases and walk parameters
T_ds = 0.2
T_ss = 0.8
N_ds = int(T_ds / dt)
N_ss = int(T_ss / dt)
swing_apex = 0.1
step_length = args.step_length

def ztraj(swing_apex, t_ss, ts):
    return swing_apex * np.sin(ts / t_ss * np.pi)


contact_phases = (
    ["DOUBLE"] * N_ds
    + ["LEFT"] * N_ss
    + ["DOUBLE"] * N_ds
)

LF_placements = []
RF_placements = []
nsteps = len(contact_phases)
current_x = RF_placement.translation[0]

ts = 0
for cp in contact_phases:
    ts += 1
    if cp == "DOUBLE":
        ts = 0
        RF_goal = RF_placement.copy()
        RF_goal.translation[0] = current_x
        LF_placements.append(LF_placement)
        RF_placements.append(RF_goal)
    # if cp == "RIGHT":
    #     LF_goal = LF_placement.copy()
    #     LF_goal.translation[2] = ztraj(swing_apex, N_ss, ts)
    #     LF_placements.append(LF_goal)
    #     RF_placements.append(RF_placement)
    if cp == "LEFT":
        RF_goal = RF_placement.copy()
        current_x += step_length / N_ss
        RF_goal.translation[0] = current_x
        RF_goal.translation[2] = ztraj(swing_apex, N_ss, ts)
        LF_placements.append(LF_placement)
        RF_placements.append(RF_goal)
        
stages = [createStage(contact_phases[0], "DOUBLE", LF_placements[0], RF_placements[0])]
for i in range(1, nsteps):
    stages.append(
        createStage(
            contact_phases[i], contact_phases[i - 1], LF_placements[i], RF_placements[i]
        )
    )

problem = aligator.TrajOptProblem(x0, stages, term_cost)

TOL = 1e-4
mu_init = 1e-8
rho_init = 0.0
# verbose = aligator.VerboseLevel.VERBOSE
verbose = aligator.VerboseLevel.QUIET
solve_times = []
infeas = []
for iter in range(1, args.max_iters + 1):
    print(iter)
    solver = aligator.SolverProxDDP(TOL, mu_init, rho_init, verbose=verbose)
    # solver = aligator.SolverFDDP(TOL, verbose=verbose)
    solver.rollout_type = aligator.ROLLOUT_LINEAR
    # solver = aligator.SolverFDDP(TOL, verbose=verbose)
    solver.max_iters = iter
    solver.sa_strategy = aligator.SA_FILTER  # FILTER or LINESEARCH
    solver.filter.beta = 1e-5
    solver.force_initial_condition = True
    solver.reg_min = 1e-6
    solver.linear_solver_choice = aligator.LQ_SOLVER_PARALLEL  # LQ_SOLVER_SERIAL
    solver.setNumThreads(args.num_threads)
    solver.setup(problem)

    us_init = [np.zeros(nu)] * nsteps
    xs_init = [x0] * (nsteps + 1)

    tic = time.time()
    solver.run(
        problem,
        xs_init,
        us_init,
    )
    toc = time.time()
    solve_time = toc - tic
    # print("Elapsed time: ", solve_time)
    workspace = solver.workspace
    results = solver.results
    # print(results)
    solve_times.extend([solve_time])
    infeas.extend([results.primal_infeas])
    
print(args.step_length, args.dt)
scipy.io.savemat('data/talos_walk_single_step_speed_test_' + str(args.step_length) + '_' + str(args.dt) + '.mat', 
                 {'solve_times': solve_times, 'infeas': infeas})

# # save the results
# xs_opt = np.asarray(results.xs.tolist())
# us_opt = np.asarray(results.us.tolist())

# ## forward simulation with integrator_floatingbase_helper
# dt_sim = 5e-4

# # note that we only focus on the first left support stage
# constraint_model_left = [constraint_models[0]]
# constraint_data_left = [constraint_datas[0]]

# ts_left = np.linspace(0, T_ss, N_ss)
# xs_left = np.array(xs_opt[N_ds : N_ds + N_ss])
# us_left = np.array(us_opt[N_ds : N_ds + N_ss])
# x0 = xs_left[0]

# # evaluation of the optimal control solution from aligator
# Kp = np.diag(60.0 * np.ones(nv))
# # Kd = np.diag(0.05 * np.sqrt(60.0) * np.ones(nv))
# Kd = np.diag(5.0 * np.ones(nv))

# ts_sim, sol, us_sim, e_sim, edot_sim = integrator_floatingbase_helper.integrate(
#     rmodel, constraint_model_left, constraint_data_left,
#     0, T_ss, dt_sim, x0,
#     ts_left, xs_left, us_left, act_matrix,
#     Kp, Kd)

# pin.forwardKinematics(rmodel, rdata, xs_opt[-1][:nq])
# pin.updateFramePlacements(rmodel, rdata)
# RF_placement = rdata.oMf[RF_id]
# step_length_opt = RF_placement.translation[0] - RF_placements[0].translation[0]
# pin.forwardKinematics(rmodel, rdata, sol[-1][:nq])
# pin.updateFramePlacements(rmodel, rdata)
# RF_placement = rdata.oMf[RF_id]
# step_length_sim = RF_placement.translation[0] - RF_placements[0].translation[0]
# print(step_length_opt, step_length_sim)

# result_filename = 'data/talos_walk_single_step_aligator_' + str(args.step_length) + '_' + str(args.dt) + '.mat'
# scipy.io.savemat(result_filename, {'ts_sim': ts_sim, \
#                                    'sol': sol, \
#                                    'us_sim': us_sim, \
#                                    'e_sim': e_sim, \
#                                    'edot_sim': edot_sim, \
#                                    'step_length_sim': step_length_sim, \
#                                    'xs_opt': xs_left, \
#                                    'us_opt': us_left, \
#                                    'step_length_opt': step_length_opt, \
#                                    'solve_time': solve_time})


# def fdisplay():
#     # qs = [x[:nq] for x in results.xs.tolist()]
#     qs = [x[:nq] for x in sol]

#     for _ in range(5):
#         vizer.play(qs, dt)
#         time.sleep(0.5)


# if args.display:
#     # vizer.setCameraPosition([1.2, 0.0, 1.2])
#     # vizer.setCameraTarget([0.0, 0.0, 1.0])
#     fdisplay()
