import numpy as np
import math
import pinocchio as pin
from scipy.interpolate import interp1d

def integrate(model, constraint_model, constraint_data,
              t0, tf, dt, x0, 
              ts = None, xs = None, us = None, act_matrix = None,
              Kp = None, Kd = None,
              kind='zero'):
    t = 0.0
    nq = model.nq
    nv = model.nv
    data_sim = model.createData()
    pin.initConstraintDynamics(model, data_sim, constraint_model)
    prox_settings = pin.ProximalSettings(1e-12, 1e-12, 10)
    
    def control(t, q, v):
        u_openloop = interp1d(ts, us.T, kind=kind)(t)
        
        xdes = interp1d(ts, xs.T, kind=kind)(t)
        qdes = xdes[:nq]
        vdes = xdes[nq:]
        
        e = qdes - q
        edot = vdes - v
        
        e = e[1:] # remove floating base so that the dimension is consistent
        e[:6] = 0.0 # remove floating base so that the dimension is consistent
        
        u_feedback = act_matrix.T @ (Kp @ e + Kd @ edot)
        
        return u_openloop + u_feedback, e, edot

    def dynamics(t, q, v):
        if ts is None:
            tau = np.zeros((nv))
        else:
            u, _, _ = control(t, q, v)
            tau = act_matrix @ u
        
        a = pin.constraintDynamics(
            model, data_sim, q, v, tau, constraint_model, constraint_data, prox_settings
        )
        
        # print(t)
        # for i, cd in enumerate(constraint_data):
        #     print(i, cd.c1Mc2.translation.T)
        # print(' ')
        
        return a
    
    t = t0
    q = x0[:nq]
    v = x0[nq:]
    
    ts_sim = [t0]
    sol = [x0]
    
    while t <= tf:
        a = dynamics(t, q, v)
        
        v += a * dt
        q = pin.integrate(model, q, v * dt)
        t += dt

        ts_sim.append(t)
        sol.append(np.concatenate((q, v)))
    
    controls = []
    position_errors = []
    velocity_errors = []
    
    if ts is not None:
        for i in range(len(ts_sim) - 1):
            t = ts_sim[i]
            x = sol[i]
            u, e, edot = control(t, x[:nq], x[nq:])
            
            controls.append(u)
            position_errors.append(e)
            velocity_errors.append(edot)
    
    return ts_sim, sol, controls, position_errors, velocity_errors