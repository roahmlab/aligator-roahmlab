import aligator
import numpy as np
import time
import math
import sys

import pinocchio as pin
import example_robot_data

from aligator import constraints, manifolds, dynamics  # noqa
from pinocchio.visualize import MeshcatVisualizer

from utils import load_digit, ArgsBase, integrator_floatingbase_helper

import scipy.io


robot, qref = load_digit()
rmodel: pin.Model = robot.model
rdata: pin.Data = robot.data
nq = rmodel.nq
nv = rmodel.nv
nu = nv - 6

q0 = qref
v0 = np.zeros((nv))
x0 = np.concatenate([q0, v0])

# Start a new MeshCat server and client.
# Note: the server can also be started separately using the "meshcat-server" command in a terminal:
# this enables the server to remain active after the current script ends.
#
# Option open=True pens the visualizer.
# Note: the visualizer can also be opened seperately by visiting the provided URL.
try:
    viz = MeshcatVisualizer(rmodel, robot.collision_model, robot.visual_model)
    viz.initViewer(open=True)
except ImportError as err:
    print(
        "Error while initializing the viewer. It seems you should install Python meshcat"
    )
    print(err)
    sys.exit(0)

# Load the robot in the viewer.
viz.loadViewerModel()

pin.forwardKinematics(rmodel, rdata, q0)
pin.updateFramePlacements(rmodel, rdata)

constraint_models = []

# left toe A closed loop
pl1 = pin.SE3.Identity()
pl1.translation = np.array([0.17 * 2, 0, 0])
pl2 = pin.SE3.Identity()
pl2.translation = np.array([0.0179, -0.009551, -0.054164])
contact_model_ltA = pin.RigidConstraintModel(
    pin.ContactType.CONTACT_3D,   
    rmodel,
    rmodel.getJointId('left_A2_joint'),
    pl1,
    rmodel.getJointId('left_toe_roll_joint'),
    pl2,
)
contact_model_ltA.corrector.Kp[:] = (100, 100, 100)
contact_model_ltA.corrector.Kd[:] = (10, 10, 10)
constraint_models.extend([contact_model_ltA])

# left toe B closed loop
pl1 = pin.SE3.Identity()
pl1.translation = np.array([0.144 * 2, 0, 0])
pl2 = pin.SE3.Identity()
pl2.translation = np.array([-0.0181, -0.009551, -0.054164])
contact_model_ltB = pin.RigidConstraintModel(
    pin.ContactType.CONTACT_3D,   
    rmodel,
    rmodel.getJointId('left_B2_joint'),
    pl1,
    rmodel.getJointId('left_toe_roll_joint'),
    pl2,
)
contact_model_ltB.corrector.Kp[:] = (100, 100, 100)
contact_model_ltB.corrector.Kd[:] = (10, 10, 10)
constraint_models.extend([contact_model_ltB])

# left knee-tarsus closed loop
pl1 = pin.SE3.Identity()
pl1.translation = np.array([0.25 * 2, 0, 0])

# heel spring transformation
pl2_1 = pin.SE3.Identity()
pl2_1.rotation = pin.rpy.rpyToMatrix(np.array([math.radians(4.47), math.radians(0.32), math.radians(155.8)]))
pl2_1.translation = np.array([-0.01766, -0.029456, 0.00104])

pl2_2 = pin.SE3.Identity()
pl2_2.translation = np.array([0.113789, -0.011056, 0])

pl2 = pl2_1 * pl2_2
contact_model_lkt = pin.RigidConstraintModel(
    pin.ContactType.CONTACT_3D,   
    rmodel,
    rmodel.getJointId('left_ach2_joint'),
    pl1,
    rmodel.getJointId('left_tarsus_joint'),
    pl2,
)
contact_model_lkt.corrector.Kp[:] = (100, 100, 100)
contact_model_lkt.corrector.Kd[:] = (10, 10, 10)
constraint_models.extend([contact_model_lkt])

# right toe A closed loop
pl1 = pin.SE3.Identity()
pl1.translation = np.array([0.17 * 2, 0, 0])
pl2 = pin.SE3.Identity()
pl2.translation = np.array([0.0179, 0.009551, -0.054164])
contact_model_rtA = pin.RigidConstraintModel(
    pin.ContactType.CONTACT_3D,   
    rmodel,
    rmodel.getJointId('right_A2_joint'),
    pl1,
    rmodel.getJointId('right_toe_roll_joint'),
    pl2,
)
contact_model_rtA.corrector.Kp[:] = (100, 100, 100)
contact_model_rtA.corrector.Kd[:] = (10, 10, 10)
constraint_models.extend([contact_model_rtA])

# right toe B closed loop
pl1 = pin.SE3.Identity()
pl1.translation = np.array([0.144 * 2, 0, 0])
pl2 = pin.SE3.Identity()
pl2.translation = np.array([-0.0181, 0.009551, -0.054164])
contact_model_rtB = pin.RigidConstraintModel(
    pin.ContactType.CONTACT_3D,   
    rmodel,
    rmodel.getJointId('right_B2_joint'),
    pl1,
    rmodel.getJointId('right_toe_roll_joint'),
    pl2,
)
contact_model_rtB.corrector.Kp[:] = (100, 100, 100)
contact_model_rtB.corrector.Kd[:] = (10, 10, 10)
constraint_models.extend([contact_model_rtB])

# right knee-tarsus closed loop
pl1 = pin.SE3.Identity()
pl1.translation = np.array([0.25 * 2, 0, 0])
pl2_1 = pin.SE3.Identity()
pl2_1.rotation = pin.rpy.rpyToMatrix(np.array([math.radians(-4.47), math.radians(0.32), math.radians(-155.8)]))
pl2_1.translation = np.array([-0.01766, 0.029456, 0.00104])

pl2_2 = pin.SE3.Identity()
pl2_2.translation = np.array([0.113789, 0.011056, 0])

pl2 = pl2_1 * pl2_2
contact_model_rkt = pin.RigidConstraintModel(
    pin.ContactType.CONTACT_3D,   
    rmodel,
    rmodel.getJointId('right_ach2_joint'),
    pl1,
    rmodel.getJointId('right_tarsus_joint'),
    pl2,
)
contact_model_rkt.corrector.Kp[:] = (100, 100, 100)
contact_model_rkt.corrector.Kd[:] = (10, 10, 10)
constraint_models.extend([contact_model_rkt])

# left foot contact
contact_model_lfc = pin.RigidConstraintModel(
    pin.ContactType.CONTACT_6D,   
    rmodel,
    rmodel.getJointId('left_toe_roll_joint'),
    robot.pl_leftfoot,
    0,
    rdata.oMf[rmodel.getFrameId('left_foot')],
    pin.LOCAL_WORLD_ALIGNED
)
contact_model_lfc.corrector.Kp[:] = (0, 0, 100, 0, 0, 0)
contact_model_lfc.corrector.Kd[:] = (50, 50, 50, 50, 50, 50)
constraint_models.extend([contact_model_lfc])

# right foot contact
contact_model_rfc = pin.RigidConstraintModel(
    pin.ContactType.CONTACT_6D,   
    rmodel,
    rmodel.getJointId('right_toe_roll_joint'),
    robot.pl_rightfoot,
    0,
    rdata.oMf[rmodel.getFrameId('right_foot')],
    pin.LOCAL_WORLD_ALIGNED
)
contact_model_rfc.corrector.Kp[:] = (0, 0, 100, 0, 0, 0)
contact_model_rfc.corrector.Kd[:] = (50, 50, 50, 50, 50, 50)
constraint_models.extend([contact_model_rfc])

viz.display(q0)

constraint_datas = [cm.createData() for cm in constraint_models]

# naive test without any control
dt_sim = 5e-4
ts_sim, sol, us_sim, e_sim, edot_sim = integrator_floatingbase_helper.integrate(
    rmodel, constraint_models, constraint_datas,
    0, 0.3, dt_sim, x0)

qs = [x[:nq] for x in sol]

for _ in range(5):
    viz.play(qs, dt_sim)
    time.sleep(0.5)