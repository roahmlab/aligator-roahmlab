import numpy as np
import aligator
import pinocchio as pin
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import time

from aligator import (
    manifolds,
    dynamics,
    constraints,
)
from utils import load_talos_no_wristhead, ArgsBase, derivQuat


class Args(ArgsBase):
    tcp: str = None
    bounds: bool = True
    num_threads: int = 8


args = Args().parse_args()
robotComplete, robot = load_talos_no_wristhead()
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

pin.forwardKinematics(rmodel, rdata, q0)
pin.updateFramePlacements(rmodel, rdata)

space = manifolds.MultibodyPhaseSpace(rmodel)

x0 = np.concatenate((q0, np.zeros(nv)))
u0 = np.zeros(nu)
dt = 1e-3
duration = 0.3

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
    # cm.corrector.Kp[:] = (20, 20, 20, 20, 20, 20)
    # cm.corrector.Kd[:] = (2, 2, 2, 2, 2, 2)
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
    return ode, dyn_model

ode, dyn_model = create_dynamics("LEFT")

def dyn(t, x, ode):
    d = ode.createData()
    u = np.zeros(nu)
    ode.forward(x, u, d)
    
    # time derivative of a quaternion given angular velocity
    # technically we shouldn't do that, the integration of quaternion should be handled directly inside the integrator
    # but since we are using scipy's solve_ivp, we have to do this trick
    # we normalize the quaternion to avoid numerical issues after solve_ivp
    # check https://github.com/google-deepmind/mujoco/blob/main/src/engine/engine_util_spatial.c#L243 
    # or https://arxiv.org/abs/1711.02508 Eq 183
    # on how to integrate quaternion properly
    dx = np.zeros(nq + nv)
    dx[:3] = d.xdot[:3]
    dx[3:7] = derivQuat(x[3:7], d.xdot[3:6])
    dx[7:] = d.xdot[6:]
    
    return dx

ts = np.arange(0.0, duration, dt)
solution = solve_ivp(lambda t, x: dyn(t, x, ode),
                     [ts[0], ts[-1]],
                     x0, 
                     method='RK45', 
                     t_eval=ts)

# normalize the quaternion in the solution
for i in range(1, len(ts)):
    solution.y[3:7, i] /= np.linalg.norm(solution.y[3:7, i])
    

def fdisplay():
    qs = [x[:nq] for x in solution.y.T]

    for _ in range(5):
        vizer.play(qs, dt)
        time.sleep(0.5)


if args.display:
    # vizer.setCameraPosition([1.2, 0.0, 1.2])
    # vizer.setCameraTarget([0.0, 0.0, 1.0])
    fdisplay()
