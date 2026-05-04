"""
utils.py
Conversión cartesiano ↔ kepleriano, energía orbital y utilidades.
"""
import numpy as np
from astropy.time import Time
from .constants import GM_SOL_AU3_DAY2

def state_to_orbital_elements(pos, vel, gm=GM_SOL_AU3_DAY2):
    r, v = np.linalg.norm(pos), np.linalg.norm(vel)
    E = 0.5*v**2 - gm/r
    a = -gm/(2*E) if abs(E) > 1e-15 else np.inf
    h = np.cross(pos, vel)
    h_norm = np.linalg.norm(h)
    i = np.degrees(np.arccos(np.clip(h[2]/h_norm, -1, 1)))
    
    k = np.array([0,0,1])
    n_vec = np.cross(k, h)
    n_norm = np.linalg.norm(n_vec)
    Omega = np.degrees(np.arccos(np.clip(n_vec[0]/n_norm, -1, 1))) if n_norm > 1e-15 else 0.0
    if n_norm > 1e-15 and n_vec[1] < 0: Omega = 360 - Omega
    
    e_vec = ((v**2 - gm/r)*pos - np.dot(pos, vel)*vel)/gm
    e = np.linalg.norm(e_vec)
    omega = np.degrees(np.arccos(np.clip(np.dot(n_vec, e_vec)/(n_norm*e), -1, 1))) if n_norm > 1e-15 and e > 1e-10 else 0.0
    if n_norm > 1e-15 and e > 1e-10 and e_vec[2] < 0: omega = 360 - omega
    
    nu = np.degrees(np.arccos(np.clip(np.dot(e_vec, pos)/(e*r), -1, 1))) if e > 1e-10 else 0.0
    if e > 1e-10 and np.dot(pos, vel) < 0: nu = 360 - nu
    
    E_anom = np.degrees(np.arccos(np.clip((e + np.cos(np.radians(nu)))/(1 + e*np.cos(np.radians(nu))), -1, 1))) if e < 1.0 else 0.0
    if e < 1.0 and nu > 180: E_anom = 360 - E_anom
    M = (E_anom - np.degrees(e * np.sin(np.radians(E_anom)))) % 360 if e < 1.0 else 0.0
    
    return {"a": float(a), "e": float(e), "i": i, "omega": omega, "Omega": Omega, "M": M, "E": E}

def semi_major_axis(pos, vel, gm=GM_SOL_AU3_DAY2):
    E = 0.5*np.linalg.norm(vel)**2 - gm/np.linalg.norm(pos)
    return -gm/(2*E) if abs(E) > 1e-15 else np.inf

def check_energy_conservation(result, tol=1e-6):
    pos, vel = result["asteroid_pos"], result["asteroid_vel"]
    E = 0.5*np.sum(vel**2, axis=1) - GM_SOL_AU3_DAY2/np.linalg.norm(pos, axis=1)
    dE = (E.max() - E.min()) / abs(E[0]) if abs(E[0]) > 1e-15 else 0.0
    return {"passed": dE < tol, "variation_rel": float(dE), "E_initial": float(E[0]), "E_final": float(E[-1])}

def jd_to_iso(jd): return Time(jd, format="jd", scale="tdb").iso[:10]
def iso_to_jd(s): return float(Time(s, format="iso", scale="tdb").jd)
def au_to_km(au): return au * 1.495978707e8
def au_to_ld(au): return au_to_km(au) / 384400.0