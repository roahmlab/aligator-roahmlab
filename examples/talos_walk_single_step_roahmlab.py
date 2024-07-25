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
from utils import load_talos_only_legs, ArgsBase


class Args(ArgsBase):
    tcp: str = None
    bounds: bool = True
    num_threads: int = 8
    max_iters: int = 200


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
# q0 = rmodel.referenceConfigurations["half_sitting"]
v0 = np.zeros(nv)

# q0 = np.array(
#     [
#         -0.167777,
#         -0.00143857,
#         1.04461,
#         0.025096255649710599161528179479319,
#         0.00029357227751950908285760721838642,
#         -0.010878593912774356736172798321149,
#         0.99962580396974698437873030343326,
#         0.0217575,
#         -0.194412,
#         -0.332625,
#         0.141048,
#         0.191535,
#         0.144211,
#         0.0217574,
#         -0.196618,
#         0.388312,
#         -0.182366,
#         -0.205988,
#         0.146417
#     ]
# )
q0 = np.array(
    [
        -0.0689173,
        -0.0317775,
        1.07092,
        0.010948527018694632212403305970838, 
        -0.000046386116997883724893600443373032, 
        -0.0030930082910206970793487446513836,
        0.99993527835763484912234844159684, 
        0.00620187,
        -0.106981,
        -0.197428,
        0.155264,
        0.0423248,
        0.0850842,
        0.00620198,
        -0.107501,
        0.238134,
        -0.173242,
        -0.0647299,
        0.0856035
    ]
)

pin.forwardKinematics(rmodel, rdata, q0)
pin.updateFramePlacements(rmodel, rdata)

space = manifolds.MultibodyPhaseSpace(rmodel)

x0 = np.concatenate((q0, v0))
u0 = np.zeros(nu)
com0 = pin.centerOfMass(rmodel, rdata, x0[:nq])
dt = 0.01

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
    umax = rmodel.effortLimit[6:]
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
T_ds = 20
T_ss = 80
swing_apex = 0.2
step_length = 0.2

def ztraj(swing_apex, t_ss, ts):
    return swing_apex * np.sin(ts / t_ss * np.pi)


contact_phases = (
    ["DOUBLE"] * T_ds
    + ["LEFT"] * T_ss
    + ["DOUBLE"] * T_ds
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
    if cp == "RIGHT":
        LF_goal = LF_placement.copy()
        LF_goal.translation[2] = ztraj(swing_apex, T_ss, ts)
        LF_placements.append(LF_goal)
        RF_placements.append(RF_placement)
    if cp == "LEFT":
        RF_goal = RF_placement.copy()
        current_x += 2.0 * step_length / T_ss
        RF_goal.translation[0] = current_x
        RF_goal.translation[2] = ztraj(swing_apex, T_ss, ts)
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
solver = aligator.SolverProxDDP(TOL, mu_init, rho_init, verbose=verbose)
# solver = aligator.SolverFDDP(TOL, verbose=verbose)
solver.rollout_type = aligator.ROLLOUT_LINEAR
# solver = aligator.SolverFDDP(TOL, verbose=verbose)
solver.max_iters = args.max_iters
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
print("Elapsed time: ", solve_time)
workspace = solver.workspace
results = solver.results
print(results)

# save the results
xs = results.xs.tolist()
us = results.us.tolist()

qs = [x[:nq] for x in xs]
vs = [x[nq:] for x in xs]

# scipy.io.savemat("data/talos_walk_single_step_roahmlab_trajectory_" + str(args.max_iters) + ".mat", {"qs": qs, "vs": vs, "us": us})

traj_cost = results.traj_cost
prim_infeas = results.primal_infeas
# scipy.io.savemat("data/talos_walk_single_step_roahmlab_result_" + str(args.max_iters) + ".mat", {"traj_cost": traj_cost, "prim_infeas": prim_infeas, "solve_time": solve_time})
file_path = 'data/recorded_data.txt'
if args.max_iters == 1:
    with open(file_path, 'w') as file:
        file.write("%d %f %f %f\n"%(args.max_iters, traj_cost, prim_infeas, solve_time))
else:
    with open(file_path, 'a') as file:
        file.write("%d %f %f %f\n"%(args.max_iters, traj_cost, prim_infeas, solve_time))
        
    

# analyze solution
# for q in qs:
#     pin.forwardKinematics(rmodel, rdata, q0)
#     pin.updateFramePlacements(rmodel, rdata)
#     LF_placement = rdata.oMf[LF_id]
#     RF_placement = rdata.oMf[RF_id]
    
#     print(LF_placement)


def fdisplay():
    qs = [x[:nq] for x in results.xs.tolist()]

    for _ in range(10):
        vizer.play(qs, dt)
        time.sleep(0.5)


if args.display:
    # vizer.setCameraPosition([1.2, 0.0, 1.2])
    # vizer.setCameraTarget([0.0, 0.0, 1.0])
    fdisplay()
