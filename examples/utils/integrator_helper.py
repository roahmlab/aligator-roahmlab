import numpy as np
import math
import pinocchio as pin
from scipy.interpolate import interp1d
from scipy.integrate import solve_ivp

def integrate(model, ts_sim, x0, 
              ts = None, xs = None, us = None, act_matrix = None,
              Kp = None, Kd = None,
              kind='zero',
              method='RK45'):
    t = 0.0
    nq = model.nq
    nv = model.nv
    data_sim = model.createData()
    
    def control(t, x):
        u_openloop = interp1d(ts, us.T, kind=kind)(t)
        
        xdes = interp1d(ts, xs[:-1,:].T, kind=kind)(t)
        qdes = xdes[:nq]
        vdes = xdes[nq:]
        
        q = x[:nq]
        v = x[nq:]
        
        e = qdes - q
        edot = vdes - v
        
        u_feedback = Kp @ e + Kd @ edot
        
        return u_openloop + u_feedback, e, edot

    def dynamics(t, x):
        q = x[:nq]
        v = x[nq:]
        
        if ts is None:
            tau = np.zeros((nv))
        else:
            u, _, _ = control(t, x)
            tau = act_matrix @ u
        
        a = pin.aba(model, data_sim, q, v, tau)
        
        return np.concatenate([v, a])
    
    sol = solve_ivp(dynamics, 
                    [ts_sim[0], ts_sim[-1]],
                    x0, 
                    method=method,
                    t_eval=ts_sim)
    
    controls = []
    position_errors = []
    velocity_errors = []
    
    for i in range(len(ts_sim)):
        t = ts_sim[i]
        x = sol.y[:,i]
        u, e, edot = control(t, x)
        
        controls.append(u)
        position_errors.append(e)
        velocity_errors.append(edot)
    
    return sol.y.T, controls, position_errors, velocity_errors