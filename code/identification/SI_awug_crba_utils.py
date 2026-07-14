from __future__ import annotations


import copy
import json
import math
import os
import re
import sys
import time
from copy import deepcopy
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy.optimize import lsq_linear
from torch.func import functional_call

WING_FORCE_PARAM_DIM = 4
WING_MOMENT_PARAM_DIM = 7
WING_HYDRO_PARAM_DIM = WING_FORCE_PARAM_DIM + WING_MOMENT_PARAM_DIM


# =============================================================================
# 1. Default physical parameters
# =============================================================================
def _vec3(x: float, y: float, z: float, *, scale: float = 1.0) -> np.ndarray:
    """Small helper for column-vector parameters stored in millimeter/meter units."""
    return np.array([[x], [y], [z]], dtype=np.float64) * scale


def build_default_params() -> dict:
    """Build the nominal parameter dictionary used before any stage updates."""
    params = {
        # Wing hinge and buoyancy-center geometry.
        "r_lh": _vec3(90.72, -50.05, 0.0, scale=1e-3),
        "ld": 109.280e-3,
        "r_lB": _vec3(0.0, -109.280, 0.0, scale=1e-3),
        "r_lG_ne": _vec3(0.0, 11.752, 0.0, scale=1e-3),
        "r_lG": _vec3(0.0, 11.752 - 109.280, 0.0, scale=1e-3),
        "r_rh": _vec3(90.72, 50.05, 0.0, scale=1e-3),
        "rd": 109.280e-3,
        "r_rB": _vec3(0.0, 109.280, 0.0, scale=1e-3),
        "r_rG_ne": _vec3(0.0, -11.752, 0.0, scale=1e-3),
        "r_rG": _vec3(0.0, -11.752 + 109.280, 0.0, scale=1e-3),

        # Fuselage CG/CB, masses and buoyancy.
        "r_bG": _vec3(0.481, 0.0, 6.264, scale=1e-3),
        "r_bB": _vec3(0.0, 0.0, 0.0, scale=1e-3),
        "m_b": 5.590 - 68.4e-3,
        "B_b": 6.511 * 9.8,
        "m_l": 0.119,
        "B_l": 1.1662,
        "m_r": 0.119,
        "B_r": 1.1662,
        "g": 9.8,

        # Rigid-body inertia.
        "I_b": np.array(
            [
                [16963.852, -3.622, 3325.399],
                [-3.622, 114539.676, 6.582],
                [3325.399, 6.582, 112412.783],
            ],
            dtype=np.float64,
        )
        * 1e-6,
        "I_l": np.array(
            [
                [670.8795, -0.4585, 0.0],
                [-0.4585, 22.0480, 0.0030],
                [0.0, 0.0030, 690.3735],
            ],
            dtype=np.float64,
        )
        * 1e-6,
        "I_r": np.array(
            [
                [670.8795, -0.4585, 0.0],
                [-0.4585, 22.0480, 0.0030],
                [0.0, 0.0030, 690.3735],
            ],
            dtype=np.float64,
        )
        * 1e-6,

        # Environment.
        "rho": 1000.0,

        # Internal moving masses.
        "m1": 414.9e-3,
        "m2": 463.2e-3,
        "l1": 6.41e-3,
        "l2": 39.315e-3,
        "l3_zero": -30.09e-3,

        # Ballast/piston subsystem.
        "m3": 68.4e-3,
        "kappa_water": 1.0 / 0.6,
        "l_piston": 105.41e-3,
        "l_z": 46.61e-3,
        "l_ref": 128.112e-3,

        # Wing geometry.
        "S": 0.20 * 0.05,
        "l": 0.20,
        "c": 0.05,
        "r": 0.025,
        "k": _vec3(0.0, 0.0, 1.0),
        "i": _vec3(1.0, 0.0, 0.0),
        "I_3": np.eye(3, dtype=np.float64),
    }

    # Added-mass defaults from potential-flow approximation.
    m_drainage = 6.604
    params["m_drainage"] = m_drainage
    params["M_ab"] = np.array(
        [
            [0.109 * m_drainage, 0.0, 0.0],
            [0.0, 0.821 * m_drainage, 0.0],
            [0.0, 0.0, 0.821 * m_drainage],
        ],
        dtype=np.float64,
    )
    I_ref_pitch_yaw = m_drainage * (0.2565**2 + 0.075**2) / 5.0
    params["I_ab"] = np.array(
        [
            [0.01 * I_ref_pitch_yaw, 0.0, 0.0],
            [0.0, 0.62 * I_ref_pitch_yaw, 0.0],
            [0.0, 0.0, 0.62 * I_ref_pitch_yaw],
        ],
        dtype=np.float64,
    )
    # params["I_ab"] = np.array(
    #     [
    #         [0.01 * m_drainage, 0.0, 0.0],
    #         [0.0, 0.62 * m_drainage, 0.0],
    #         [0.0, 0.0, 0.62 * m_drainage],
    #     ],
    #     dtype=np.float64,
    # )

    a = 0.025
    b = 0.006
    span = params["l"]
    rho = params["rho"]
    m_added_chord = rho * np.pi * b**2 * span
    m_added_normal = rho * np.pi * a**2 * span
    # Finite-span correction: spanwise added mass should stay small but nonzero,
    # otherwise ratio-box training keeps it locked at zero.  The spanwise term
    # is weaker than the normal added mass because flow can escape around the
    # wing tips; keep this ratio conservative and tune it as a nominal prior.
    span_added_mass_ratio = 0.02
    m_added_span = min(span_added_mass_ratio * m_added_normal, m_added_chord)
    params["M_al"] = np.array(
        [
            [m_added_chord, 0.0, 0.0],
            [0.0, m_added_span, 0.0],
            [0.0, 0.0, m_added_normal],
        ],
        dtype=np.float64,
    )
    params["I_al"] = np.array(
        [
            [params["M_al"][2, 2] * span**2 / 12.0, 0.0, 0.0],
            [0.0, rho * np.pi * (a**2 - b**2) ** 2 * span / 8.0, 0.0],
            [0.0, 0.0, params["M_al"][0, 0] * span**2 / 12.0],
        ],
        dtype=np.float64,
    )
    params["M_ar"] = params["M_al"]
    params["I_ar"] = params["I_al"]
    return params


mwauv_params = build_default_params()


# =============================================================================
# 2. Spatial algebra helpers
# =============================================================================
def _as_col3(v: torch.Tensor) -> torch.Tensor:
    """Return a `(3, 1)` column view."""
    return v.reshape(3, 1)


def inertial_to_body_matrix(e: torch.Tensor) -> torch.Tensor:
    """ZYX Euler rotation from inertial frame to body frame."""
    phi, theta, psi = _as_col3(e).reshape(3)

    cphi, sphi = torch.cos(phi), torch.sin(phi)
    cth, sth = torch.cos(theta), torch.sin(theta)
    cps, sps = torch.cos(psi), torch.sin(psi)

    row1 = torch.stack([cth * cps, cth * sps, -sth])
    row2 = torch.stack([sphi * sth * cps - cphi * sps, sphi * sth * sps + cphi * cps, sphi * cth])
    row3 = torch.stack([cphi * sth * cps + sphi * sps, cphi * sth * sps - sphi * cps, cphi * cth])
    return torch.stack([row1, row2, row3], dim=0)


def skew(v: torch.Tensor) -> torch.Tensor:
    """Skew-symmetric matrix `[v]x` for a 3D vector."""
    x, y, z = _as_col3(v).reshape(3)
    zero = torch.zeros((), dtype=v.dtype, device=v.device)
    row1 = torch.stack([zero, -z, y])
    row2 = torch.stack([z, zero, -x])
    row3 = torch.stack([-y, x, zero])
    return torch.stack([row1, row2, row3], dim=0)


def ad_operator(nu: torch.Tensor) -> torch.Tensor:
    """Kinematic adjoint operator `ad_nu` in spatial vector form."""
    v = _as_col3(nu[:3])
    w = _as_col3(nu[3:])
    sw = skew(w)
    sv = skew(v)
    top = torch.cat([sw, sv], dim=1)
    bottom = torch.cat([torch.zeros_like(sw), sw], dim=1)
    return torch.cat([top, bottom], dim=0)


def ad_star(nu: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
    """Co-adjoint operator action `ad*_nu(h)`."""
    return ad_operator(nu).T @ h


def rotz(theta: torch.Tensor) -> torch.Tensor:
    """Rotation matrix around z-axis."""
    c = torch.cos(theta)
    s = torch.sin(theta)
    z = torch.zeros_like(theta)
    o = torch.ones_like(theta)
    row1 = torch.stack([c, -s, z])
    row2 = torch.stack([s, c, z])
    row3 = torch.stack([z, z, o])
    return torch.stack([row1, row2, row3], dim=0)


def roty(alpha: torch.Tensor) -> torch.Tensor:
    """Rotation matrix around y-axis."""
    c = torch.cos(alpha)
    s = torch.sin(alpha)
    z = torch.zeros_like(alpha)
    o = torch.ones_like(alpha)
    row1 = torch.stack([c, z, s])
    row2 = torch.stack([z, o, z])
    row3 = torch.stack([-s, z, c])
    return torch.stack([row1, row2, row3], dim=0)


def spatial_force_transform(R_k_to_b: torch.Tensor, r_kb_b: torch.Tensor) -> torch.Tensor:
    """Spatial force transform matrix `X` that maps wrench from frame `k` to `b`."""
    z3 = torch.zeros((3, 3), dtype=R_k_to_b.dtype, device=R_k_to_b.device)
    top = torch.cat([R_k_to_b, z3], dim=1)
    bottom = torch.cat([skew(r_kb_b) @ R_k_to_b, R_k_to_b], dim=1)
    return torch.cat([top, bottom], dim=0)


def spatial_inertia_rigid(m: torch.Tensor, I: torch.Tensor, r_g: torch.Tensor) -> torch.Tensor:
    """Rigid-body spatial inertia expressed at a shifted reference point."""
    I3 = torch.eye(3, dtype=I.dtype, device=I.device)
    S = skew(r_g)
    top = torch.cat([m * I3, -m * S], dim=1)
    bottom = torch.cat([m * S, I], dim=1)
    return torch.cat([top, bottom], dim=0)


def spatial_inertia_point_mass(m: torch.Tensor, r: torch.Tensor) -> torch.Tensor:
    """Point-mass spatial inertia expressed at a shifted reference point."""
    I3 = torch.eye(3, dtype=r.dtype, device=r.device)
    S = skew(r)
    top = torch.cat([m * I3, -m * S], dim=1)
    bottom = torch.cat([m * S, -m * S @ S], dim=1)
    return torch.cat([top, bottom], dim=0)


def phi_transform(r: torch.Tensor) -> torch.Tensor:
    """Velocity-shift transform `Phi(r)` used by the parallel-axis operation."""
    I3 = torch.eye(3, dtype=r.dtype, device=r.device)
    Z3 = torch.zeros((3, 3), dtype=r.dtype, device=r.device)
    return torch.cat([torch.cat([I3, -skew(r)], dim=1), torch.cat([Z3, I3], dim=1)], dim=0)


# =============================================================================
# 3. AWUG dynamics model
# =============================================================================
class AWUGModelFromTex(nn.Module):
    """Hybrid underwater glider model used by the SI pipeline."""

    def __init__(
        self,
        params: Dict | None = None,
        device: str = "cpu",
        dtype: torch.dtype = torch.double,
        train_body_added_mass_inertia: bool | None = None,
        train_body_hydro_params: bool | None = None,
        train_wing_added_mass_inertia: bool | None = None,
        train_wing_hydro_params: bool | None = None,
        train_wing_added_mass_inertia_gates: bool | None = None,
        train_wing_hydro_gates: bool | None = None,
        freeze_k2: bool = False,
        enable_sweep_gates: bool = True,
        enable_pressure_center_migration: bool = True,
        enable_added_mass_update: bool = True,
        freeze_sweep_geometry: bool = False,
        freeze_added_mass_scaling: bool = False,
        decouple_gate_eta_from_geometry_freeze: bool = False,
    ) -> None:
        super().__init__()
        self.device = device
        self.dtype = dtype
        self.p = mwauv_params if params is None else params

        train_body_added_mass_inertia = bool(train_body_added_mass_inertia)
        train_body_hydro_params = bool(train_body_hydro_params)
        train_wing_added_mass_inertia = bool(train_wing_added_mass_inertia)
        train_wing_hydro_params = bool(train_wing_hydro_params)
        train_wing_added_mass_inertia_gates = bool(train_wing_added_mass_inertia_gates)
        train_wing_hydro_gates = bool(train_wing_hydro_gates)
        self.enable_sweep_gates = bool(enable_sweep_gates)
        self.enable_pressure_center_migration = bool(enable_pressure_center_migration)
        self.enable_added_mass_update = bool(enable_added_mass_update)
        self.freeze_sweep_geometry = bool(freeze_sweep_geometry)
        self.freeze_added_mass_scaling = bool(freeze_added_mass_scaling)
        self.decouple_gate_eta_from_geometry_freeze = bool(decouple_gate_eta_from_geometry_freeze)

        def _to_arr(value, shape: tuple[int, ...] | None = None) -> np.ndarray:
            """Convert config values to float64 numpy arrays with optional reshape."""
            arr = np.asarray(value, dtype=np.float64)
            if shape is not None:
                arr = arr.reshape(shape)
            return arr

        def _coerce_theta6(value) -> np.ndarray:
            """Map legacy 5-term wing moment parameters into the current 7-term order."""
            arr = np.asarray(value, dtype=np.float64).reshape(-1, 1)
            if arr.shape[0] == WING_MOMENT_PARAM_DIM:
                return arr
            if arr.shape[0] == 5:
                migrated = np.zeros((WING_MOMENT_PARAM_DIM, 1), dtype=np.float64)
                # Legacy order: [C_lb, C_lp, C_m0, C_ma, C_nb].
                # Current order: [C_xbeta, C_xp_d, C_m0, C_malpha, C_mq_d, C_zbeta, C_zr_d].
                migrated[0] = arr[0]
                migrated[1] = arr[1]
                migrated[2] = arr[2]
                migrated[3] = arr[3]
                migrated[5] = arr[4]
                return migrated
            raise ValueError(
                f"Theta_6 expects {WING_MOMENT_PARAM_DIM} entries "
                f"(or legacy 5 entries), got {arr.shape[0]}"
            )

        def _buf(name: str, value, shape: tuple[int, ...] | None = None) -> None:
            """Register a non-trainable tensor constant."""
            self.register_buffer(
                name,
                torch.tensor(_to_arr(value, shape), dtype=dtype, device=device),
            )

        def _param(
            key: str,
            fallback,
            *,
            trainable: bool,
            shape: tuple[int, ...] | None = None,
        ) -> nn.Parameter:
            """Create a model parameter from params[key] or fallback."""
            value = self.p.get(key, fallback)
            if key == "Theta_6":
                value = _coerce_theta6(value)
            arr = _to_arr(value, shape)
            return nn.Parameter(torch.tensor(arr, dtype=dtype, device=device), requires_grad=trainable)

        def _scalar_param(value: float, trainable: bool) -> nn.Parameter:
            """Create a scalar trainable/non-trainable parameter."""
            return nn.Parameter(torch.tensor(float(value), dtype=dtype, device=device), requires_grad=trainable)

        def _gate_paramdict(
            key: str,
            names: Iterable[str],
            trainable: bool,
            default: float = 1.0,
        ) -> nn.ParameterDict:
            """Create a ParameterDict for named gate coefficients."""
            src = self.p.get(key, {})
            if not isinstance(src, dict):
                src = {}
            return nn.ParameterDict(
                {n: _scalar_param(float(src.get(n, default)), trainable) for n in names}
            )

        _buf("g0", self.p["g"])
        _buf("k_axis", self.p["k"], (3, 1))

        _buf("m_b", self.p["m_b"])
        _buf("B_b", self.p["B_b"])
        _buf("r_bB", self.p["r_bB"], (3, 1))
        _buf("r_bG", self.p["r_bG"], (3, 1))
        _buf("I_b", self.p["I_b"], (3, 3))

        self.M_ab = _param("M_ab", self.p["M_ab"], trainable=train_body_added_mass_inertia, shape=(3, 3))
        self.I_ab = _param("I_ab", self.p["I_ab"], trainable=train_body_added_mass_inertia, shape=(3, 3))

        _buf("m_l", self.p["m_l"])
        _buf("B_l", self.p["B_l"])
        _buf("r_lG", self.p["r_lG"], (3, 1))
        _buf("r_lh", self.p["r_lh"], (3, 1))
        _buf("r_lB", self.p["r_lB"], (3, 1))
        _buf("I_l", self.p["I_l"], (3, 3))

        _buf("m_r", self.p["m_r"])
        _buf("B_r", self.p["B_r"])
        _buf("r_rG", self.p["r_rG"], (3, 1))
        _buf("r_rh", self.p["r_rh"], (3, 1))
        _buf("r_rB", self.p["r_rB"], (3, 1))
        _buf("I_r", self.p["I_r"], (3, 3))

        self.M_al = _param("M_al", self.p["M_al"], trainable=train_wing_added_mass_inertia, shape=(3, 3))
        self.I_al = _param("I_al", self.p["I_al"], trainable=train_wing_added_mass_inertia, shape=(3, 3))
        # Keep explicit aliases used by some legacy call-sites.
        self.M_ar = self.M_al
        self.I_ar = self.I_al

        _buf("m1", self.p["m1"])
        _buf("m2", self.p["m2"])
        _buf("m3_base", self.p["m3"])
        _buf("rho", self.p["rho"])
        _buf("l1", self.p["l1"])
        _buf("l2", self.p["l2"])
        _buf("l3_zero", self.p["l3_zero"])
        _buf("wing_b0", self.p["l"])
        _buf("wing_c0", self.p["c"])
        _buf("wing_r", self.p["r"])
        _buf("kappa_water", self.p["kappa_water"])
        _buf("l_piston", self.p["l_piston"])
        _buf("l_z", self.p["l_z"])
        _buf("l_ref", self.p["l_ref"])

        _buf("r_dL", [[0.0], [-float(self.p["ld"])], [0.0]], (3, 1))
        _buf("r_dR", [[0.0], [float(self.p["rd"])], [0.0]], (3, 1))

        gate_added_mass_defaults = {
            "k_XA_x": 0.0,
            "k_XA_y": 0.0,
            "k_XA_z": 0.0,
            "k_XA_x2": 0.0,
            "k_XA_y2": 0.0,
            "k_XA_z2": 0.0,
            "k_KA": 0.0,
            "k_MA": 0.0,
            "k_NA": 0.0,
            "k_KA2": 0.0,
            "k_MA2": 0.0,
            "k_NA2": 0.0,
        }
        for name, default in gate_added_mass_defaults.items():
            init = float(self.p.get(name, default))
            setattr(self, name, _scalar_param(init, train_wing_added_mass_inertia_gates))

        self.k_g2_X5 = _gate_paramdict(
            "k_g2_X5",
            (
                "C_D0_k1",
                "C_Da2_k1",
                "C_Yb_k1",
                "C_La_k1",
                "C_D0_k2",
                "C_Da2_k2",
                "C_Yb_k2",
                "C_La_k2",
            ),
            train_wing_hydro_gates,
            default=0.0,
        )
        self.k_g2_X6 = _gate_paramdict(
            "k_g2_X6",
            (
                "C_xbeta_k1",
                "C_xp_d_k1",
                "C_m0_k1",
                "C_malpha_k1",
                "C_mq_d_k1",
                "C_zbeta_k1",
                "C_zr_d_k1",
                "C_xbeta_k2",
                "C_xp_d_k2",
                "C_m0_k2",
                "C_malpha_k2",
                "C_mq_d_k2",
                "C_zbeta_k2",
                "C_zr_d_k2",
            ),
            train_wing_hydro_gates,
            default=0.0,
        )

        if freeze_k2:
            for p in (self.k_XA_x2, self.k_XA_y2, self.k_XA_z2, self.k_KA2, self.k_MA2, self.k_NA2):
                p.data.zero_()
                p.requires_grad_(False)
            for name, p in self.k_g2_X5.items():
                if name.endswith("_k2"):
                    p.data.zero_()
                    p.requires_grad_(False)
            for name, p in self.k_g2_X6.items():
                if name.endswith("_k2"):
                    p.data.zero_()
                    p.requires_grad_(False)

        self.Theta_3 = _param(
            "Theta_3",
            [[-5.0], [-20.0], [-8.0], [-30.0], [-1.0], [-3.0], [-8.0], [-30.0], [-1.0], [-3.0]],
            trainable=train_body_hydro_params,
            shape=(10, 1),
        )
        self.Theta_4 = _param(
            "Theta_4",
            [[-0.5], [-1.5], [-1.0], [-3.0], [-0.5], [-2.0], [-1.0], [-3.0], [-0.5], [-2.0]],
            trainable=train_body_hydro_params,
            shape=(10, 1),
        )
        self.Theta_5 = _param(
            "Theta_5",
            [[0.2], [1.0], [0.2], [1.2]],
            trainable=train_wing_hydro_params,
            shape=(WING_FORCE_PARAM_DIM, 1),
        )
        self.Theta_6 = _param(
            "Theta_6",
            [[0.05], [0.02], [0.0], [-0.3], [0.02], [0.05], [0.02]],
            trainable=train_wing_hydro_params,
            shape=(WING_MOMENT_PARAM_DIM, 1),
        )

    @staticmethod
    def _as_scalar(x: torch.Tensor) -> torch.Tensor:
        """Return a 0-D tensor view."""
        return x.reshape(())

    @staticmethod
    def _gate_sweep(eta: torch.Tensor, a: torch.Tensor, b: torch.Tensor | None = None) -> torch.Tensor:
        # Sweep gate: K(1)=1, K(0)=1-a.  The k2 term changes only the interior
        # curvature, so k2=0 exactly recovers the legacy first-order gate.
        second_order = 0.0 if b is None else b * eta * (eta - 1.0)
        return 1.0 + a * (eta - 1.0) + second_order

    @staticmethod
    def _gate_added_mass(eta: torch.Tensor, a: torch.Tensor, b: torch.Tensor | None = None) -> torch.Tensor:
        """Gate for sweep-dependent added-mass scaling."""
        # Added-mass gate: K(0)=0, K(1)=1.  The k2 basis is independent from
        # the k1 curvature term while preserving those endpoints.
        eta_one_minus_eta = eta * (1.0 - eta)
        second_order = 0.0 if b is None else b * eta_one_minus_eta * (2.0 * eta - 1.0)
        return eta + a * eta_one_minus_eta + second_order

    @staticmethod
    def _apply_body_symmetry(theta_full: torch.Tensor) -> torch.Tensor:
        """Apply paired body symmetry using z-direction coefficients as source."""
        # Original y-source mapping:
        # red = theta_full[:6]
        # return torch.cat([red, red[2:4], -red[4:6]], dim=0)
        x = theta_full[:2]
        z_src = theta_full[6:10]
        return torch.cat([x, z_src[:2], -z_src[2:4], z_src], dim=0)

    @staticmethod
    def _construct_X3_matrix(v_b: torch.Tensor, w_b: torch.Tensor, device: str, dtype: torch.dtype) -> torch.Tensor:
        """Construct the fuselage force regressor for Theta_3."""
        u, v, w = v_b[0:1], v_b[1:2], v_b[2:3]
        q, r = w_b[1:2], w_b[2:3]
        z = torch.zeros((1, 1), dtype=dtype, device=device)
        row1 = torch.cat([u, u * torch.abs(u), z, z, z, z, z, z, z, z], dim=1)
        row2 = torch.cat([z, z, v, v * torch.abs(v), r, r * torch.abs(r), z, z, z, z], dim=1)
        row3 = torch.cat([z, z, z, z, z, z, w, w * torch.abs(w), q, q * torch.abs(q)], dim=1)
        return torch.cat([row1, row2, row3], dim=0)

    @staticmethod
    def _construct_X4_matrix(v_b: torch.Tensor, w_b: torch.Tensor, device: str, dtype: torch.dtype) -> torch.Tensor:
        """Construct the fuselage moment regressor for Theta_4."""
        v, w = v_b[1:2], v_b[2:3]
        p, q, r = w_b[0:1], w_b[1:2], w_b[2:3]
        z = torch.zeros((1, 1), dtype=dtype, device=device)
        row1 = torch.cat([p, p * torch.abs(p), z, z, z, z, z, z, z, z], dim=1)
        row2 = torch.cat([z, z, q, q * torch.abs(q), w, w * torch.abs(w), z, z, z, z], dim=1)
        row3 = torch.cat([z, z, z, z, z, z, r, r * torch.abs(r), v, v * torch.abs(v)], dim=1)
        return torch.cat([row1, row2, row3], dim=0)

    def _b_eff(self, theta_abs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Effective wing span ratio with optional geometry freeze."""
        theta_abs = torch.abs(theta_abs)
        b = self.wing_b0
        if self.freeze_sweep_geometry:
            ones = torch.ones_like(theta_abs)
            b_eff_const = ones * b
            return ones, b_eff_const
        c = self.wing_c0
        r = self.wing_r
        b_cal = b + r
        b_eff = b_cal * torch.cos(theta_abs) + 0.5 * c * torch.sin(theta_abs) - 0.5 * c
        eta_b = b_eff / (b + 1e-10)
        return eta_b, b_eff

    def _b_eff_unfrozen(self, theta_abs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Effective wing span ratio without freeze, used by decoupled gate terms."""
        theta_abs = torch.abs(theta_abs)
        b = self.wing_b0
        c = self.wing_c0
        r = self.wing_r
        b_cal = b + r
        b_eff = b_cal * torch.cos(theta_abs) + 0.5 * c * torch.sin(theta_abs) - 0.5 * c
        eta_b = b_eff / (b + 1e-10)
        return eta_b, b_eff

    def _S_eff(self, theta_abs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Effective wing area ratio with optional geometry freeze."""
        theta_abs = torch.abs(theta_abs)
        b = self.wing_b0
        c = self.wing_c0
        if self.freeze_sweep_geometry:
            ones = torch.ones_like(theta_abs)
            s0 = b * c
            s_eff_const = ones * s0
            return ones, s_eff_const
        r = self.wing_r
        b_cal = b + r

        arg = ((b_cal / (r + 1e-10)) ** 2 - 1.0) / (1.0 + (b_cal / (r + 1e-10)) ** 2 + 1e-10)
        arg = torch.clamp(arg, -1.0, 1.0)
        theta_c = torch.asin(arg)

        eps = torch.tensor(1e-10, dtype=self.dtype, device=self.device)
        cos_t = torch.cos(theta_abs)
        sin_t = torch.sin(theta_abs)
        tan_t = torch.tan(theta_abs)

        S_a = c * (b_cal - r / torch.clamp(cos_t, min=eps))
        xi_star = (b_cal * cos_t - r) / torch.clamp(sin_t, min=eps)
        S_b = 0.5 * tan_t * (xi_star + c / 2.0) ** 2
        S_eff = torch.where(theta_abs <= theta_c, S_a, S_b)

        S0 = c * b
        eta_s = torch.clamp(S_eff / (S0 + 1e-10), 0.0, 1.0)
        return eta_s, S_eff

    def _S_eff_unfrozen(self, theta_abs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Effective wing area ratio without geometry freeze."""
        theta_abs = torch.abs(theta_abs)
        b = self.wing_b0
        c = self.wing_c0
        r = self.wing_r
        b_cal = b + r

        arg = ((b_cal / (r + 1e-10)) ** 2 - 1.0) / (1.0 + (b_cal / (r + 1e-10)) ** 2 + 1e-10)
        arg = torch.clamp(arg, -1.0, 1.0)
        theta_c = torch.asin(arg)

        eps = torch.tensor(1e-10, dtype=self.dtype, device=self.device)
        cos_t = torch.cos(theta_abs)
        sin_t = torch.sin(theta_abs)
        tan_t = torch.tan(theta_abs)

        S_a = c * (b_cal - r / torch.clamp(cos_t, min=eps))
        xi_star = (b_cal * cos_t - r) / torch.clamp(sin_t, min=eps)
        S_b = 0.5 * tan_t * (xi_star + c / 2.0) ** 2
        S_eff = torch.where(theta_abs <= theta_c, S_a, S_b)

        S0 = c * b
        eta_s = torch.clamp(S_eff / (S0 + 1e-10), 0.0, 1.0)
        return eta_s, S_eff

    def _gate_eta_terms(self, theta_abs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return area/span-derived eta terms used by sweep and added-mass gates."""
        theta_abs = torch.abs(theta_abs)
        if self.decouple_gate_eta_from_geometry_freeze and self.freeze_sweep_geometry:
            eta_a, _ = self._S_eff_unfrozen(theta_abs)
            eta_b, _ = self._b_eff_unfrozen(theta_abs)
        else:
            eta_a, _ = self._S_eff(theta_abs)
            eta_b, _ = self._b_eff(theta_abs)
        eta_kn = eta_a * (eta_b**2)
        return eta_a, eta_b, eta_kn

    def _wing_added_mass_gates(self, theta_k: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute sweep-dependent scaling for wing added mass and inertia."""
        theta_abs = torch.abs(theta_k)
        eta_s, _, eta_kn = self._gate_eta_terms(theta_abs)

        if self.freeze_added_mass_scaling:
            ones = torch.ones_like(eta_s)
            g_ma = torch.stack([ones, ones, ones])
            g_ia = torch.stack([ones, ones, ones])
            return g_ma.reshape(3), g_ia.reshape(3)

        if not self.enable_added_mass_update:
            g_ma = torch.stack([eta_s, eta_s, eta_s])
            g_ia = torch.stack([eta_kn, eta_s, eta_kn])
            return g_ma.reshape(3), g_ia.reshape(3)

        g_ma = torch.stack(
            [
                self._gate_added_mass(eta_s, self.k_XA_x, self.k_XA_x2),
                self._gate_added_mass(eta_s, self.k_XA_y, self.k_XA_y2),
                self._gate_added_mass(eta_s, self.k_XA_z, self.k_XA_z2),
            ]
        )
        g_ia = torch.stack(
            [
                self._gate_added_mass(eta_kn, self.k_KA, self.k_KA2),
                self._gate_added_mass(eta_s, self.k_MA, self.k_MA2),
                self._gate_added_mass(eta_kn, self.k_NA, self.k_NA2),
            ]
        )
        return g_ma.reshape(3), g_ia.reshape(3)

    def _r_cp(self, theta_abs: torch.Tensor, wing: str) -> torch.Tensor:
        """Compute pressure-center migration point for a wing."""
        theta_abs = torch.abs(theta_abs)
        if not self.enable_pressure_center_migration:
            theta_abs = torch.zeros_like(theta_abs)
        b = self.wing_b0
        c = self.wing_c0
        r = self.wing_r
        b_cal = b + r

        arg = ((b_cal / (r + 1e-10)) ** 2 - 1.0) / (1.0 + (b_cal / (r + 1e-10)) ** 2 + 1e-10)
        arg = torch.clamp(arg, -1.0, 1.0)
        theta_c = torch.asin(arg)

        eps = torch.tensor(1e-10, dtype=self.dtype, device=self.device)
        cos_t = torch.cos(theta_abs)
        sin_t = torch.sin(theta_abs)
        tan_t = torch.tan(theta_abs)

        denom_a = torch.clamp(b_cal - r / torch.clamp(cos_t, min=eps), min=eps)
        xi_a = -(c**2 * tan_t) / (12.0 * denom_a)
        eta_a = 0.5 * (b_cal + r / torch.clamp(cos_t, min=eps)) - (c**2 * tan_t**2) / (24.0 * denom_a)

        xi_b = ((b_cal * cos_t - r) / torch.clamp(sin_t, min=eps) - c) / 3.0
        eta_b = ((r - 0.5 * c * sin_t) / torch.clamp(cos_t, min=eps) + 2.0 * b_cal) / 3.0

        xi = torch.where(theta_abs <= theta_c, xi_a, xi_b)
        eta = torch.where(theta_abs <= theta_c, eta_a, eta_b)

        z = torch.zeros_like(xi)
        if wing == "L":
            return torch.stack([-xi, -eta, z]).reshape(3, 1)
        if wing == "R":
            return torch.stack([-xi, eta, z]).reshape(3, 1)
        return torch.zeros((3, 1), dtype=self.dtype, device=self.device)

    def _R_wa(self, alpha_k: torch.Tensor, beta_k: torch.Tensor) -> torch.Tensor:
        """Rotation from wind/aero axes to the wing local frame.

        The local flow direction is parameterized as
        ``[cos(alpha)cos(beta), sin(beta), sin(alpha)cos(beta)]`` because
        ``alpha = atan2(v_z, v_x)`` and ``beta = asin(v_y / V)`` in this
        model.  Therefore the first wind-axis basis vector must align with the
        local velocity direction, so ``R_wa @ [-D, 0, 0]`` is opposite to the
        incoming flow.
        """
        ca = torch.cos(alpha_k)
        sa = torch.sin(alpha_k)
        cb = torch.cos(beta_k)
        sb = torch.sin(beta_k)
        z = torch.zeros((), dtype=self.dtype, device=self.device)
        return torch.stack(
            [
                torch.stack([ca * cb, -ca * sb, -sa]),
                torch.stack([sb, cb, z]),
                torch.stack([sa * cb, -sa * sb, ca]),
            ]
        )

    def _construct_X5_matrix(
        self,
        theta_k: torch.Tensor,
        alpha_k: torch.Tensor,
        beta_k: torch.Tensor,
        V_k: torch.Tensor,
        A_k: torch.Tensor,
    ) -> torch.Tensor:
        """Construct the wing force coefficient regressor for Theta_5."""
        q_dyn_area = 0.5 * self.rho * (V_k**2) * A_k
        eta_a, _, _ = self._gate_eta_terms(theta_k)

        if self.enable_sweep_gates:
            g_cd0 = self._gate_sweep(eta_a, self.k_g2_X5["C_D0_k1"], self.k_g2_X5["C_D0_k2"])
            g_cda2 = self._gate_sweep(eta_a, self.k_g2_X5["C_Da2_k1"], self.k_g2_X5["C_Da2_k2"])
            g_cyb = self._gate_sweep(eta_a, self.k_g2_X5["C_Yb_k1"], self.k_g2_X5["C_Yb_k2"])
            g_cla = self._gate_sweep(eta_a, self.k_g2_X5["C_La_k1"], self.k_g2_X5["C_La_k2"])
        else:
            g_cd0 = torch.ones_like(eta_a)
            g_cda2 = torch.ones_like(eta_a)
            g_cyb = torch.ones_like(eta_a)
            g_cla = torch.ones_like(eta_a)

        # Final single-wing finite-angle basis (still linear in Theta_5):
        # D = q*S*(C_D0 + Delta_C_Dalpha*sin^2(alpha))
        # S = q*S*(C_Sbeta*sin(beta)*cos(beta))
        # L = q*S*(C_Lalpha*sin(alpha)*cos(alpha))
        sa2 = torch.sin(alpha_k) ** 2
        sin_a_cos_a = torch.sin(alpha_k) * torch.cos(alpha_k)
        sin_b_cos_b = torch.sin(beta_k) * torch.cos(beta_k)

        z = torch.zeros((1, 1), dtype=self.dtype, device=self.device)
        row1 = torch.cat(
            [
                (-q_dyn_area * g_cd0).reshape(1, 1),
                (-q_dyn_area * sa2 * g_cda2).reshape(1, 1),
                z,
                z,
            ],
            dim=1,
        )
        row2 = torch.cat([z, z, (q_dyn_area * sin_b_cos_b * g_cyb).reshape(1, 1), z], dim=1)
        row3 = torch.cat([z, z, z, (-q_dyn_area * sin_a_cos_a * g_cla).reshape(1, 1)], dim=1)
        return torch.cat([row1, row2, row3], dim=0)

    def _construct_X6_matrices(
        self,
        theta_abs: torch.Tensor,
        alpha_k: torch.Tensor,
        beta_k: torch.Tensor,
        V_k: torch.Tensor,
        A_k: torch.Tensor,
        w_k: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Construct static-aero-axis and rate-wing-axis moment regressors."""
        _, b_eff = self._b_eff(torch.abs(theta_abs))
        eta_a_gate, eta_b_gate, eta_kn_gate = self._gate_eta_terms(torch.abs(theta_abs))
        eta_sb_gate = torch.clamp(eta_a_gate * eta_b_gate, 0.0, 1.0)

        c_eff = A_k / (b_eff + 1e-10)
        q_dyn_area = 0.5 * self.rho * (V_k**2) * A_k
        p_k = w_k[0, 0]
        q_k = w_k[1, 0]
        r_k = w_k[2, 0]
        p_hat = p_k * b_eff / (2.0 * V_k + 1e-10)
        q_hat = q_k * c_eff / (2.0 * V_k + 1e-10)
        r_hat = r_k * b_eff / (2.0 * V_k + 1e-10)

        if self.enable_sweep_gates:
            g_cxb = self._gate_sweep(eta_sb_gate, self.k_g2_X6["C_xbeta_k1"], self.k_g2_X6["C_xbeta_k2"])
            g_cxp = self._gate_sweep(eta_kn_gate, self.k_g2_X6["C_xp_d_k1"], self.k_g2_X6["C_xp_d_k2"])
            g_cm0 = self._gate_sweep(eta_a_gate, self.k_g2_X6["C_m0_k1"], self.k_g2_X6["C_m0_k2"])
            g_cma = self._gate_sweep(eta_a_gate, self.k_g2_X6["C_malpha_k1"], self.k_g2_X6["C_malpha_k2"])
            g_cmq = self._gate_sweep(eta_a_gate, self.k_g2_X6["C_mq_d_k1"], self.k_g2_X6["C_mq_d_k2"])
            g_czb = self._gate_sweep(eta_sb_gate, self.k_g2_X6["C_zbeta_k1"], self.k_g2_X6["C_zbeta_k2"])
            g_czr = self._gate_sweep(eta_kn_gate, self.k_g2_X6["C_zr_d_k1"], self.k_g2_X6["C_zr_d_k2"])
        else:
            g_cxb = torch.ones_like(eta_a_gate)
            g_cxp = torch.ones_like(eta_a_gate)
            g_cm0 = torch.ones_like(eta_a_gate)
            g_cma = torch.ones_like(eta_a_gate)
            g_cmq = torch.ones_like(eta_a_gate)
            g_czb = torch.ones_like(eta_a_gate)
            g_czr = torch.ones_like(eta_a_gate)

        sin_a_cos_a = torch.sin(alpha_k) * torch.cos(alpha_k)
        sin_b_cos_b = torch.sin(beta_k) * torch.cos(beta_k)

        z = torch.zeros((1, 1), dtype=self.dtype, device=self.device)
        static_row1 = torch.cat(
            [
                (q_dyn_area * b_eff * sin_b_cos_b * g_cxb).reshape(1, 1),
                z,
                z,
                z,
                z,
                z,
                z,
            ],
            dim=1,
        )
        static_row2 = torch.cat(
            [
                z,
                z,
                (q_dyn_area * c_eff * g_cm0).reshape(1, 1),
                (q_dyn_area * c_eff * sin_a_cos_a * g_cma).reshape(1, 1),
                z,
                z,
                z,
            ],
            dim=1,
        )
        static_row3 = torch.cat(
            [
                z,
                z,
                z,
                z,
                z,
                (q_dyn_area * b_eff * sin_b_cos_b * g_czb).reshape(1, 1),
                z,
            ],
            dim=1,
        )
        rate_row1 = torch.cat(
            [
                z,
                (-q_dyn_area * b_eff * p_hat * g_cxp).reshape(1, 1),
                z,
                z,
                z,
                z,
                z,
            ],
            dim=1,
        )
        rate_row2 = torch.cat(
            [
                z,
                z,
                z,
                z,
                (-q_dyn_area * c_eff * q_hat * g_cmq).reshape(1, 1),
                z,
                z,
            ],
            dim=1,
        )
        rate_row3 = torch.cat(
            [
                z,
                z,
                z,
                z,
                z,
                z,
                (-q_dyn_area * b_eff * r_hat * g_czr).reshape(1, 1),
            ],
            dim=1,
        )
        X6_static_a = torch.cat([static_row1, static_row2, static_row3], dim=0)
        X6_rate_k = torch.cat([rate_row1, rate_row2, rate_row3], dim=0)
        return X6_static_a, X6_rate_k

    def _wing_hydrodynamic_wrench(self, nu_k: torch.Tensor, theta_abs: torch.Tensor, wing: str) -> torch.Tensor:
        """Compute local wing hydrodynamic force/moment wrench."""
        theta_abs = torch.abs(theta_abs)
        v_k = nu_k[:3]
        w_k = nu_k[3:]
        r_cp = self._r_cp(theta_abs, wing)
        v_aero = v_k + skew(w_k) @ r_cp

        V = torch.linalg.norm(v_aero) + 1e-10
        alpha_k = torch.atan2(v_aero[2, 0], v_aero[0, 0] + 1e-10)
        beta_k = torch.asin(torch.clamp(v_aero[1, 0] / V, -1.0, 1.0))
        _, s_k = self._S_eff(theta_abs)

        X5 = self._construct_X5_matrix(theta_abs, alpha_k, beta_k, V, s_k)
        X6_static_a, X6_rate_k = self._construct_X6_matrices(theta_abs, alpha_k, beta_k, V, s_k, w_k)

        R_wa = self._R_wa(alpha_k, beta_k)
        F_hydro = R_wa @ (X5 @ self.Theta_5)
        M_hydro_H = R_wa @ (X6_static_a @ self.Theta_6) + X6_rate_k @ self.Theta_6
        M_hydro_D = M_hydro_H + skew(r_cp) @ F_hydro
        return torch.cat([F_hydro, M_hydro_D], dim=0)

    def _fuselage_hydrodynamic_wrench(self, nu_b: torch.Tensor) -> torch.Tensor:
        """Compute fuselage hydrodynamic wrench from Theta_3/Theta_4."""
        v_b = nu_b[:3]
        w_b = nu_b[3:]
        X3 = self._construct_X3_matrix(v_b, w_b, self.device, self.dtype)
        X4 = self._construct_X4_matrix(v_b, w_b, self.device, self.dtype)
        F_b = X3 @ self._apply_body_symmetry(self.Theta_3)
        T_b = X4 @ self._apply_body_symmetry(self.Theta_4)
        return torch.cat([F_b, T_b], dim=0)

    def compute_tau_ext(self, nu_b: torch.Tensor, q_s: torch.Tensor, qd_s: torch.Tensor, F_p: torch.Tensor | None = None) -> torch.Tensor:
        """Compute all external hydrodynamic/propulsion wrenches."""
        theta_L, alpha_L, theta_R, alpha_R, _, _, _ = [q_s[i].reshape(1)[0] for i in range(7)]
        dtheta_L, dalpha_L, dtheta_R, dalpha_R, _, _, _ = [qd_s[i].reshape(1)[0] for i in range(7)]

        tau_prop = torch.zeros((6, 1), dtype=self.dtype, device=self.device)
        if F_p is not None:
            tau_prop[0, 0] = self._as_scalar(F_p)

        tau_ext = self._fuselage_hydrodynamic_wrench(nu_b) + tau_prop
        for wing in ("L", "R"):
            if wing == "L":
                theta_k, alpha_k, dalpha_k, dtheta_k, r_hk_B = theta_L, alpha_L, dalpha_L, dtheta_L, self.r_lh
            else:
                theta_k, alpha_k, dalpha_k, dtheta_k, r_hk_B = theta_R, alpha_R, dalpha_R, dtheta_R, self.r_rh

            R_K2B, J_k, _ = self._wing_terms(theta_k, alpha_k, dalpha_k)
            P_k = spatial_force_transform(R_K2B, r_hk_B)
            qd_k = torch.stack([dtheta_k, dalpha_k]).reshape(2, 1)
            nu_k = P_k.T @ nu_b + J_k @ qd_k
            tau_ext = tau_ext + P_k @ self._wing_hydrodynamic_wrench(nu_k, torch.abs(theta_k), wing)
        return tau_ext

    def _build_body_inertia(self) -> torch.Tensor:
        """Build body rigid + added spatial inertia."""
        M_rb = spatial_inertia_rigid(self.m_b, self.I_b, self.r_bG)
        M_add = torch.zeros((6, 6), dtype=self.dtype, device=self.device)
        M_add[:3, :3] = self.M_ab
        M_add[3:, 3:] = self.I_ab
        return M_rb + M_add

    def _build_wing_inertia(self, wing: str, theta_k: torch.Tensor) -> torch.Tensor:
        """Build one wing's rigid + added spatial inertia."""
        if wing == "L":
            m, I, rG, Ma, Ia = self.m_l, self.I_l, self.r_lG, self.M_al, self.I_al
        else:
            m, I, rG, Ma, Ia = self.m_r, self.I_r, self.r_rG, self.M_ar, self.I_ar

        g_ma, g_ia = self._wing_added_mass_gates(theta_k)
        Ma_eff = torch.diag(g_ma) @ Ma
        Ia_eff = torch.diag(g_ia) @ Ia

        z3 = torch.zeros((3, 3), dtype=self.dtype, device=self.device)
        M_G = torch.cat(
            [
                torch.cat([m * torch.eye(3, dtype=self.dtype, device=self.device), z3], dim=1),
                torch.cat([z3, I], dim=1),
            ],
            dim=0,
        )
        M_D = phi_transform(rG).T @ M_G @ phi_transform(rG)

        M_H = torch.cat([torch.cat([Ma_eff, z3], dim=1), torch.cat([z3, Ia_eff], dim=1)], dim=0)
        r_cp = self._r_cp(theta_k, wing)
        M_D = M_D + phi_transform(r_cp).T @ M_H @ phi_transform(r_cp)
        return M_D

    def _wing_terms(
        self,
        theta_k: torch.Tensor,
        alpha_k: torch.Tensor,
        alpha_k_dot: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return wing transform, joint Jacobian, and Jacobian derivative."""
        R_K2B = rotz(theta_k) @ roty(alpha_k)
        R_B2K = R_K2B.T

        j_k = torch.tensor([[0.0], [1.0], [0.0]], dtype=self.dtype, device=self.device)
        J_omega_k = torch.cat([R_B2K @ self.k_axis, j_k], dim=1)
        J_k = torch.cat([torch.zeros((3, 2), dtype=self.dtype, device=self.device), J_omega_k], dim=0)

        col1_dot = -alpha_k_dot * (skew(j_k) @ (R_B2K @ self.k_axis))
        J_omega_k_dot = torch.cat([col1_dot, torch.zeros((3, 1), dtype=self.dtype, device=self.device)], dim=1)
        J_k_dot = torch.cat([torch.zeros((3, 2), dtype=self.dtype, device=self.device), J_omega_k_dot], dim=0)
        return R_K2B, J_k, J_k_dot

    def _point_mass_kinematics(
        self,
        theta_2: torch.Tensor,
        l_3: torch.Tensor,
        l: torch.Tensor,
    ) -> tuple[torch.Tensor, list[torch.Tensor], list[torch.Tensor]]:
        """Compute internal point-mass positions and Jacobians."""
        s2 = torch.sin(theta_2)
        c2 = torch.cos(theta_2)

        m_water = self.kappa_water * l
        m_3 = self.m3_base + m_water

        num = 0.5 * self.kappa_water * (l**2) + self.m3_base * l + 0.5 * self.m3_base * self.l_piston
        r_3x = num / (m_3 + 1e-10)
        l4 = self.l_ref - r_3x

        dr3x_dl = (
            0.5 * (self.kappa_water**2) * (l**2)
            + self.kappa_water * self.m3_base * (l - 0.5 * self.l_piston)
            + self.m3_base**2
        ) / (m_3**2 + 1e-10)
        partial_l4_l = -dr3x_dl

        r_2 = torch.stack([self.l1 - l_3, -self.l2 * s2, self.l2 * c2]).reshape(3, 1)
        r_1 = torch.stack([self.l1, -self.l2 * s2, self.l2 * c2]).reshape(3, 1)
        r_3 = torch.stack(
            [
                l4.reshape(()),
                torch.zeros((), dtype=self.dtype, device=self.device),
                self.l_z.reshape(()),
            ]
        ).reshape(3, 1)

        z = torch.zeros((), dtype=self.dtype, device=self.device)
        o = torch.ones((), dtype=self.dtype, device=self.device)
        J_m2 = torch.stack(
            [
                torch.stack([z, -o, z]),
                torch.stack([-self.l2 * c2, z, z]),
                torch.stack([-self.l2 * s2, z, z]),
            ]
        )
        J_m1 = torch.stack(
            [
                torch.stack([z, z, z]),
                torch.stack([-self.l2 * c2, z, z]),
                torch.stack([-self.l2 * s2, z, z]),
            ]
        )
        J_m3 = torch.stack(
            [
                torch.stack([z, z, partial_l4_l]),
                torch.stack([z, z, z]),
                torch.stack([z, z, z]),
            ]
        )
        return m_3, [r_1, r_2, r_3], [J_m1, J_m2, J_m3]

    @staticmethod
    def _unpack_q(
        q_s: torch.Tensor,
        qd_s: torch.Tensor,
        qdd_s: torch.Tensor,
    ) -> tuple[tuple[torch.Tensor, ...], tuple[torch.Tensor, ...], tuple[torch.Tensor, ...]]:
        """Unpack generalized coordinates and their derivatives into tuples."""
        q = tuple(q_s[i].reshape(1)[0] for i in range(7))
        qd = tuple(qd_s[i].reshape(1)[0] for i in range(7))
        qdd = tuple(qdd_s[i].reshape(1)[0] for i in range(7))
        return q, qd, qdd

    def _accumulate_terms(
        self,
        nu_b: torch.Tensor,
        eta: torch.Tensor,
        q_s: torch.Tensor,
        qd_s: torch.Tensor,
        qdd_s: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Accumulate inertia, Coriolis, internal acceleration, bias, and gravity terms."""
        (theta_L, alpha_L, theta_R, alpha_R, theta_2, l_3, l), (
            dtheta_L,
            dalpha_L,
            dtheta_R,
            dalpha_R,
            dtheta_2,
            dl_3,
            dl,
        ), (
            ddtheta_L,
            ddalpha_L,
            ddtheta_R,
            ddalpha_R,
            ddtheta_2,
            ddl_3,
            ddl,
        ) = self._unpack_q(q_s, qd_s, qdd_s)

        M_q_s = self._build_body_inertia()
        C_nu = -ad_star(nu_b, M_q_s @ nu_b)
        M_bs_qdd = torch.zeros((6, 1), dtype=self.dtype, device=self.device)
        b_s = torch.zeros((6, 1), dtype=self.dtype, device=self.device)
        g_q_s_eta = torch.zeros((6, 1), dtype=self.dtype, device=self.device)

        R_I2B = inertial_to_body_matrix(eta.reshape(3, 1))
        k_g_B = R_I2B @ torch.tensor([[0.0], [0.0], [1.0]], dtype=self.dtype, device=self.device)
        g_body = torch.cat(
            [
                -(self.m_b * self.g0 - self.B_b) * k_g_B,
                -(self.m_b * self.g0 * skew(self.r_bG) - self.B_b * skew(self.r_bB)) @ k_g_B,
            ],
            dim=0,
        )
        g_q_s_eta = g_q_s_eta + g_body

        for side in ("L", "R"):
            if side == "L":
                theta_k, alpha_k = theta_L, alpha_L
                dtheta_k, dalpha_k = dtheta_L, dalpha_L
                ddtheta_k, ddalpha_k = ddtheta_L, ddalpha_L
                r_hk_B, m_k, B_k, r_kG, r_kB = self.r_lh, self.m_l, self.B_l, self.r_lG, self.r_lB
            else:
                theta_k, alpha_k = theta_R, alpha_R
                dtheta_k, dalpha_k = dtheta_R, dalpha_R
                ddtheta_k, ddalpha_k = ddtheta_R, ddalpha_R
                r_hk_B, m_k, B_k, r_kG, r_kB = self.r_rh, self.m_r, self.B_r, self.r_rG, self.r_rB

            R_K2B, J_k, J_k_dot = self._wing_terms(theta_k, alpha_k, dalpha_k)
            P_k = spatial_force_transform(R_K2B, r_hk_B)
            M_k = self._build_wing_inertia(side, theta_k)
            qd_k = torch.tensor([[dtheta_k], [dalpha_k]], dtype=self.dtype, device=self.device)
            qdd_k = torch.tensor([[ddtheta_k], [ddalpha_k]], dtype=self.dtype, device=self.device)

            M_q_s = M_q_s + P_k @ M_k @ P_k.T
            nu_k = P_k.T @ nu_b + J_k @ qd_k
            nu_bar_k = P_k.T @ nu_b
            C_nu = C_nu + (-P_k @ ad_star(nu_bar_k, M_k @ nu_bar_k))
            M_bs_qdd = M_bs_qdd + P_k @ M_k @ J_k @ qdd_k

            sigma_k = J_k_dot @ qd_k + ad_operator(nu_k) @ (J_k @ qd_k)
            b_s = b_s + P_k @ (M_k @ sigma_k - ad_star(nu_k, M_k @ nu_k) + ad_star(nu_bar_k, M_k @ nu_bar_k))

            k_g_k = R_K2B.T @ k_g_B
            g_k = torch.cat(
                [
                    -(m_k * self.g0 - B_k) * k_g_k,
                    -(skew(m_k * self.g0 * r_kG) - B_k * skew(r_kB)) @ k_g_k,
                ],
                dim=0,
            )
            g_q_s_eta = g_q_s_eta + P_k @ g_k

        m_3, r_list, Jm_list = self._point_mass_kinematics(theta_2, l_3, l)
        m_list = [self.m1, self.m2, m_3]
        qd_int = torch.tensor([[dtheta_2], [dl_3], [dl]], dtype=self.dtype, device=self.device)
        qdd_int = torch.tensor([[ddtheta_2], [ddl_3], [ddl]], dtype=self.dtype, device=self.device)
        Jdot_qdot_12 = torch.tensor(
            [[0.0], [self.l2 * torch.sin(theta_2) * dtheta_2**2], [-self.l2 * torch.cos(theta_2) * dtheta_2**2]],
            dtype=self.dtype,
            device=self.device,
        )

        for i, (m_i, r_i, J_mi) in enumerate(zip(m_list, r_list, Jm_list)):
            M_i = spatial_inertia_point_mass(m_i, r_i)
            J_i = torch.cat([J_mi, torch.zeros((3, 3), dtype=self.dtype, device=self.device)], dim=0)
            M_q_s = M_q_s + M_i

            nu_i = nu_b + J_i @ qd_int
            nu_bar_i = nu_b
            C_nu = C_nu + (-ad_star(nu_bar_i, M_i @ nu_bar_i))
            M_bs_qdd = M_bs_qdd + M_i @ J_i @ qdd_int

            sigma_v = (Jdot_qdot_12 if i < 2 else torch.zeros((3, 1), dtype=self.dtype, device=self.device)) + skew(nu_b[3:]) @ (
                J_mi @ qd_int
            )
            sigma_i = torch.cat([sigma_v, torch.zeros((3, 1), dtype=self.dtype, device=self.device)], dim=0)
            b_s = b_s + (M_i @ sigma_i - ad_star(nu_i, M_i @ nu_i) + ad_star(nu_bar_i, M_i @ nu_bar_i))

            g_i = torch.cat([-m_i * self.g0 * k_g_B, -m_i * self.g0 * skew(r_i) @ k_g_B], dim=0)
            g_q_s_eta = g_q_s_eta + g_i

        return M_q_s, C_nu, M_bs_qdd, b_s, g_q_s_eta

    @staticmethod
    def _fp_code_to_thrust(fp_code: float | int, *, dtype: torch.dtype, device: str) -> torch.Tensor:
        """Map experiment force-code values to thrust force."""
        if int(fp_code) == 1565:
            return torch.tensor(26.77e-3 * 9.8, dtype=dtype, device=device)
        if int(fp_code) == 1560:
            return torch.tensor(25.5e-3 * 9.8, dtype=dtype, device=device)
        return torch.tensor(0.0, dtype=dtype, device=device)

    def forward(self, state: torch.Tensor, tau_ext: torch.Tensor | None = None) -> Dict[str, torch.Tensor]:
        """Evaluate body acceleration for one 39-D state row."""
        st = state.to(self.device, dtype=self.dtype).reshape(-1)

        eta = st[4:7].reshape(3, 1)
        v_b = st[7:10].reshape(3, 1)
        w_b = st[10:13].reshape(3, 1)
        nu_b = torch.cat([v_b, w_b], dim=0)

        theta_L = st[19].reshape(1)
        theta_R = st[20].reshape(1)
        alpha_L = st[25].reshape(1)
        alpha_R = st[26].reshape(1)
        dtheta_L = st[21].reshape(1)
        dtheta_R = st[22].reshape(1)
        dalpha_L = st[27].reshape(1)
        dalpha_R = st[28].reshape(1)
        ddtheta_L = st[23].reshape(1)
        ddtheta_R = st[24].reshape(1)
        ddalpha_L = st[29].reshape(1)
        ddalpha_R = st[30].reshape(1)

        theta_2 = st[31].reshape(1)
        dtheta_2 = st[32].reshape(1)
        ddtheta_2 = st[33].reshape(1)

        l_3 = (st[34] * 1e-3 - self.l3_zero + self.l1).reshape(1)
        dl_3 = (st[35] * 1e-3).reshape(1)
        ddl_3 = (st[36] * 1e-3).reshape(1)

        l = (st[37] * 1e-3 / self.kappa_water).reshape(1)
        dl = torch.zeros_like(l)
        ddl = torch.zeros_like(l)

        q_s = torch.cat([theta_L, alpha_L, theta_R, alpha_R, theta_2, l_3, l], dim=0).reshape(7, 1)
        qd_s = torch.cat([dtheta_L, dalpha_L, dtheta_R, dalpha_R, dtheta_2, dl_3, dl], dim=0).reshape(7, 1)
        qdd_s = torch.cat([ddtheta_L, ddalpha_L, ddtheta_R, ddalpha_R, ddtheta_2, ddl_3, ddl], dim=0).reshape(7, 1)

        if tau_ext is None:
            fp_code = st[38].item()
            F_p = self._fp_code_to_thrust(fp_code, dtype=self.dtype, device=self.device)
            tau_ext = self.compute_tau_ext(nu_b, q_s, qd_s, F_p=F_p)

        M_q_s, C_nu, M_bs_qdd, b_s, g_q_s_eta = self._accumulate_terms(nu_b, eta, q_s, qd_s, qdd_s)
        rhs = tau_ext - C_nu - M_bs_qdd - b_s - g_q_s_eta
        nu_b_dot = torch.linalg.solve(M_q_s, rhs)

        return {
            "M": M_q_s,
            "C_nu": C_nu,
            "M_bs_qdd_s": M_bs_qdd,
            "b_s": b_s,
            "g": g_q_s_eta,
            "nu_b_dot": nu_b_dot,
        }


if __name__ == "__main__":
    model = AWUGModelFromTex()
    dummy = torch.zeros(39, dtype=torch.double)
    out = model(dummy)
    print("nu_b_dot:", out["nu_b_dot"].reshape(-1))


# =============================================================================
# 4. Core SI utilities: data, JSON, residuals, training, plotting
# =============================================================================
class Tee:
    """Mirror writes to multiple file-like streams."""

    def __init__(self, *files):
        self.files = files

    def write(self, data):
        for f in self.files:
            f.write(data)
            f.flush()

    def flush(self):
        for f in self.files:
            f.flush()


def resolve_data_path(filename: str) -> str:
    """Resolve data filename against local `processed_data` if needed."""
    if os.path.isabs(filename) and os.path.exists(filename):
        return filename
    if os.path.exists(filename):
        return filename
    source_proj_dir = os.environ.get("SOURCE_PROJ_DIR")
    if source_proj_dir:
        candidate = os.path.join(source_proj_dir, filename)
        if os.path.exists(candidate):
            return candidate
    return os.path.join(os.path.dirname(__file__), "processed_data", filename)


def parse_files(s: str) -> List[str]:
    """Parse a comma-separated file list."""
    return [x.strip() for x in s.split(",") if x.strip()]


def parse_indexed_file_spec(spec: str) -> Tuple[str, Optional[int], Optional[int]]:
    """Parse file specs of the form path.xlsx@row_begin:row_end."""
    text = spec.strip()
    match = re.match(r"^(?P<path>.+)@(?P<begin>\d+):(?P<end>\d+)$", text)
    if not match:
        return text, None, None
    row_begin = int(match.group("begin"))
    row_end = int(match.group("end"))
    if row_end < row_begin:
        raise ValueError(f"Invalid row range in {spec!r}")
    return match.group("path"), row_begin, row_end


def read_indexed_dataframe(spec: str) -> pd.DataFrame:
    """Read one workbook and apply an optional inclusive row range."""
    filename, row_begin, row_end = parse_indexed_file_spec(spec)
    df = pd.read_excel(resolve_data_path(filename))
    if row_begin is None:
        return df
    return df.iloc[row_begin : row_end + 1].copy().reset_index(drop=True)


def read_rows(files: List[str]) -> List[pd.Series]:
    """Read all selected rows from multiple files for row-wise LS."""
    rows: List[pd.Series] = []
    for f in files:
        df = read_indexed_dataframe(f)
        for i in range(len(df)):
            rows.append(df.iloc[i, :])
    return rows


def build_state_tensor(df: pd.DataFrame, device: str, sample_step: int = 1) -> torch.Tensor:
    """Convert one experiment DataFrame into `(N, 39)` state tensor."""
    # sample_step controls downsampling; max(1, sample_step) avoids a zero stride.
    slc = df.iloc[:: max(1, sample_step), :]
    arr = np.concatenate(
        (
            slc.iloc[:, :4].to_numpy(),
            slc.iloc[:, 8:23].to_numpy(),
            slc.iloc[:, 25:45].to_numpy(),
        ),
        axis=1,
    )
    return torch.tensor(arr, dtype=torch.double, device=device)


def build_traj_data(files: List[str], device: str, sample_step: int = 1) -> List[torch.Tensor]:
    """Build trajectory tensor list from indexed workbook specs."""
    return [
        build_state_tensor(read_indexed_dataframe(f), device=device, sample_step=sample_step)
        for f in files
    ]


def row_col(row: pd.Series, name: str, default: float = 0.0) -> float:
    """Read a named row value with a numeric fallback."""
    return float(row[name]) if name in row.index else float(default)


def row_vec(row: pd.Series, names: Tuple[str, str, str]) -> torch.Tensor:
    """Read three named columns as a `(3,1)` double tensor."""
    return torch.tensor(
        [[row_col(row, names[0])], [row_col(row, names[1])], [row_col(row, names[2])]],
        dtype=torch.double,
    )


def save_json(path: str, data: Dict):
    """Save dict to JSON, converting tensors/ndarrays recursively."""

    def cvt(x):
        if isinstance(x, torch.Tensor):
            return x.detach().cpu().numpy().tolist()
        if isinstance(x, np.ndarray):
            return x.tolist()
        if isinstance(x, dict):
            return {k: cvt(v) for k, v in x.items()}
        if isinstance(x, list):
            return [cvt(v) for v in x]
        return x

    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(cvt(data), f, ensure_ascii=False, indent=2)


def load_json(path: str) -> Dict:
    """Load a UTF-8 JSON file."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_norm_scale(path: str, mode: str, device: str, dtype: torch.dtype = torch.double) -> torch.Tensor | None:
    """Load `(1,12)` scale for state channels `[1:13]`."""
    mode = (mode or "none").lower()
    if mode == "none":
        return None
    js = load_json(path)
    if mode == "minmax":
        mn = np.asarray(js["min_1_13"], dtype=np.float64).reshape(12)
        mx = np.asarray(js["max_1_13"], dtype=np.float64).reshape(12)
        scale = np.maximum(mx - mn, 1e-6)
    elif mode == "zscore":
        std = np.asarray(js["std_1_13"], dtype=np.float64).reshape(12)
        scale = np.maximum(std, 1e-6)
    else:
        raise ValueError(f"unsupported norm mode: {mode}")
    return torch.tensor(scale, dtype=dtype, device=device).reshape(1, 12)


def parse_channel_weight_spec(spec: str | None) -> np.ndarray | None:
    """Parse a comma-separated 12-channel weight specification."""
    if spec is None:
        return None
    text = str(spec).strip()
    if not text:
        return None
    vals = [float(x.strip()) for x in text.split(",") if x.strip()]
    if len(vals) != 12:
        raise ValueError(f"expected 12 task weights, got {len(vals)} from: {spec}")
    arr = np.asarray(vals, dtype=np.float64)
    arr = np.maximum(arr, 1e-12)
    return arr / np.mean(arr)


def load_loss_channel_weight(
    *,
    mode: str,
    norm_mode: str,
    device: str,
    variance_weights_json: str | None = None,
    task_weights_spec: str | None = None,
    dtype: torch.dtype = torch.double,
) -> torch.Tensor | None:
    """
    Build `(1,12)` channel weights for trajectory loss.

    Modes:
    - none: no extra weighting
    - variance: inverse-variance weights estimated from data
    - task: user-provided task weights
    - variance_task: product of both, re-normalized to mean 1
    """
    mode = (mode or "none").lower()
    if mode == "none":
        return None

    variance_arr = None
    if mode in {"variance", "variance_task"}:
        if not variance_weights_json:
            raise ValueError("loss_weight_mode requires variance_weights_json for variance-based modes")
        js = load_json(variance_weights_json)
        key = "weight_after_norm_mean1" if (norm_mode or "none").lower() != "none" else "weight_raw_mean1"
        if key not in js:
            raise KeyError(f"{variance_weights_json} missing key '{key}'")
        variance_arr = np.asarray(js[key], dtype=np.float64).reshape(12)
        variance_arr = np.maximum(variance_arr, 1e-12)
        variance_arr = variance_arr / np.mean(variance_arr)

    task_arr = None
    if mode in {"task", "variance_task"}:
        task_arr = parse_channel_weight_spec(task_weights_spec)
        if task_arr is None:
            raise ValueError("loss_weight_mode requires task_weights_spec for task-based modes")

    if mode == "variance":
        final = variance_arr
    elif mode == "task":
        final = task_arr
    elif mode == "variance_task":
        final = np.maximum(variance_arr * task_arr, 1e-12)
        final = final / np.mean(final)
    else:
        raise ValueError(f"unsupported loss weight mode: {mode}")

    return torch.tensor(final, dtype=dtype, device=device).reshape(1, 12)


def traj_loss_mse(
    preds: torch.Tensor,
    truth: torch.Tensor,
    norm_scale_1_13: torch.Tensor | None = None,
    channel_weight_1_13: torch.Tensor | None = None,
) -> torch.Tensor:
    """Trajectory MSE on state channels `[1:13]`."""
    err = preds[:, 1:13] - truth[:, 1:13]
    if norm_scale_1_13 is not None:
        err = err / (norm_scale_1_13 + 1e-12)
    err2 = err**2
    if channel_weight_1_13 is not None:
        err2 = err2 * channel_weight_1_13
    return torch.mean(err2)


def as_np(x):
    """Convert tensors or array-like objects to numpy arrays."""
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def open_stage_log(base_out_dir: str, stage_name: str, argv: List[str]):
    """Create a timestamped stage directory and tee console output to run.log."""
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(base_out_dir, stage_name, ts)
    os.makedirs(run_dir, exist_ok=True)

    log_path = os.path.join(run_dir, "run.log")
    log_file = open(log_path, "w", encoding="utf-8")
    sys.stdout = Tee(sys.__stdout__, log_file)
    sys.stderr = Tee(sys.__stderr__, log_file)

    print("=" * 80)
    print(f"{stage_name} start: {datetime.now().isoformat()}")
    print(f"cmd: {' '.join(argv)}")
    print(f"env SH_CMD: {os.environ.get('SH_CMD', '')}")
    print("=" * 80)
    return run_dir, log_path


@torch.no_grad()
def traj_mse_eval(
    model: AWUGModelFromTex,
    traj_list: List[torch.Tensor],
    dt: float,
    batch_size: int,
    batch_skip: int = 1,
    norm_scale_1_13: torch.Tensor | None = None,
    channel_weight_1_13: torch.Tensor | None = None,
    forward_fn: Optional[Callable[[torch.Tensor], Dict]] = None,
    integrator: str = "rk4",
) -> float:
    """Windowed rollout MSE with the same evaluation path as `train_gd`."""
    batches = create_traj_batches(traj_list, batch_size=batch_size, batch_skip=batch_skip)
    step_fn = get_integrator_step(integrator)
    losses = []
    for fi, bi in batches:
        bd = traj_list[fi][bi : bi + batch_size]
        if bd.shape[0] < batch_size:
            continue
        state = bd[0].clone()
        preds = []
        for k in range(batch_size - 1):
            state[13:] = bd[k, 13:]
            state = step_fn(model, state, dt, forward_fn=forward_fn)
            preds.append(state.clone())
        if not preds:
            continue
        pred = torch.stack(preds, dim=0)
        losses.append(
            float(
                traj_loss_mse(
                    pred,
                    bd[1:],
                    norm_scale_1_13=norm_scale_1_13,
                    channel_weight_1_13=channel_weight_1_13,
                ).item()
            )
        )
    return float(np.mean(losses)) if losses else 0.0


def euler_rate_from_body_omega(e: torch.Tensor, w_b: torch.Tensor) -> torch.Tensor:
    """Convert body angular velocity p/q/r to Euler angle rates."""
    phi = e[0, 0]
    theta = e[1, 0]
    # w_b = [p, q, r]。
    p, q, r = w_b[0, 0], w_b[1, 0], w_b[2, 0]
    cphi = torch.cos(phi)
    sphi = torch.sin(phi)
    cth = torch.cos(theta)
    tth = torch.tan(theta)
    return torch.tensor(
        [
            [p + q * sphi * tth + r * cphi * tth],
            [q * cphi - r * sphi],
            [q * sphi / (cth + 1e-10) + r * cphi / (cth + 1e-10)],
        ],
        dtype=torch.double,
        device=e.device,
    )


def state_derivative_from_model(
    model: AWUGModelFromTex,
    state: torch.Tensor,
    forward_fn=None,
) -> torch.Tensor:
    """Build time derivative of the 39-state vector from model acceleration."""
    st = state.reshape(-1)
    out = forward_fn(st) if forward_fn is not None else model(st)
    nu_dot = out["nu_b_dot"].reshape(6, 1)

    eta = st[4:7].reshape(3, 1)
    v_b = st[7:10].reshape(3, 1)
    w_b = st[10:13].reshape(3, 1)

    R_I2B = inertial_to_body_matrix(eta)
    p_dot = R_I2B.T @ v_b
    e_dot = euler_rate_from_body_omega(eta, w_b)

    dst = torch.zeros_like(st)
    dst[1:4] = p_dot.reshape(3)
    dst[4:7] = e_dot.reshape(3)
    dst[7:10] = nu_dot[:3, 0]
    dst[10:13] = nu_dot[3:, 0]
    return dst


def rk4_step(model: AWUGModelFromTex, state: torch.Tensor, dt: float, forward_fn=None) -> torch.Tensor:
    """Advance one integration step with fourth-order Runge-Kutta."""
    k1 = state_derivative_from_model(model, state, forward_fn=forward_fn)
    k2 = state_derivative_from_model(model, state + 0.5 * dt * k1, forward_fn=forward_fn)
    k3 = state_derivative_from_model(model, state + 0.5 * dt * k2, forward_fn=forward_fn)
    k4 = state_derivative_from_model(model, state + dt * k3, forward_fn=forward_fn)
    return state + dt * (k1 + 2 * k2 + 2 * k3 + k4) / 6.0


def rk2_step(model: AWUGModelFromTex, state: torch.Tensor, dt: float, forward_fn=None) -> torch.Tensor:
    """Advance one integration step with midpoint RK2."""
    k1 = state_derivative_from_model(model, state, forward_fn=forward_fn)
    k2 = state_derivative_from_model(model, state + 0.5 * dt * k1, forward_fn=forward_fn)
    return state + dt * k2


def get_integrator_step(integrator: str):
    """Resolve a rollout integrator name to a one-step function."""
    name = (integrator or "rk4").lower()
    if name == "rk4":
        return rk4_step
    if name == "rk2":
        return rk2_step
    raise ValueError(f"unknown integrator: {integrator!r}; expected 'rk2' or 'rk4'")


def get_current_batch_size(epoch: int, total_epochs: int, bs_min: int, bs_max: int, schedule: str) -> int:
    """Interpolate rollout window size over epochs."""
    if total_epochs <= 1:
        return int(bs_max)
    p = epoch / (total_epochs - 1)
    if schedule == "exponential":
        cur = int(bs_min * ((bs_max / max(bs_min, 1)) ** p))
    else:
        cur = int(bs_min + (bs_max - bs_min) * p)
    return max(int(bs_min), min(int(bs_max), int(cur)))


def create_traj_batches(traj_list: List[torch.Tensor], batch_size: int, batch_skip: int) -> List[Tuple[int, int]]:
    """Create `(trajectory_index, begin_index)` windows for rollout training/eval."""
    batches: List[Tuple[int, int]] = []
    step = max(1, int(batch_skip))
    for fi, traj in enumerate(traj_list):
        n = int(traj.shape[0])
        for bi in range(0, max(0, n - batch_size + 1), step):
            batches.append((fi, bi))
    return batches


def _fp_code_to_F_p(fp_code: float | int, model: AWUGModelFromTex) -> torch.Tensor:
    """Map propeller/load code in the data row to propulsion force."""
    if int(fp_code) == 1565:
        return torch.tensor(26.77e-3 * 9.8, dtype=model.dtype, device=model.device)
    if int(fp_code) == 1560:
        return torch.tensor(25.5e-3 * 9.8, dtype=model.dtype, device=model.device)
    return torch.tensor(0.0, dtype=model.dtype, device=model.device)


def _parse_state_for_residual(model: AWUGModelFromTex, st: torch.Tensor):
    """Slice one 39-D state row into variables needed by residual equations."""
    st = st.to(model.device, dtype=model.dtype).reshape(-1)

    eta = st[4:7].reshape(3, 1)
    v_b = st[7:10].reshape(3, 1)
    w_b = st[10:13].reshape(3, 1)
    nu_b = torch.cat([v_b, w_b], dim=0)

    theta_L = st[19].reshape(1)
    theta_R = st[20].reshape(1)
    alpha_L = st[25].reshape(1)
    alpha_R = st[26].reshape(1)
    dtheta_L = st[21].reshape(1)
    dtheta_R = st[22].reshape(1)
    dalpha_L = st[27].reshape(1)
    dalpha_R = st[28].reshape(1)
    ddtheta_L = st[23].reshape(1)
    ddtheta_R = st[24].reshape(1)
    ddalpha_L = st[29].reshape(1)
    ddalpha_R = st[30].reshape(1)

    theta_2 = st[31].reshape(1)
    dtheta_2 = st[32].reshape(1)
    ddtheta_2 = st[33].reshape(1)

    l_3 = (st[34] * 1e-3 - model.l3_zero + model.l1).reshape(1)
    dl_3 = (st[35] * 1e-3).reshape(1)
    ddl_3 = (st[36] * 1e-3).reshape(1)

    l = (st[37] * 1e-3 / model.kappa_water).reshape(1)
    dl = torch.zeros_like(l)
    ddl = torch.zeros_like(l)

    q_s = torch.cat([theta_L, alpha_L, theta_R, alpha_R, theta_2, l_3, l], dim=0).reshape(7, 1)
    qd_s = torch.cat([dtheta_L, dalpha_L, dtheta_R, dalpha_R, dtheta_2, dl_3, dl], dim=0).reshape(7, 1)
    qdd_s = torch.cat([ddtheta_L, ddalpha_L, ddtheta_R, ddalpha_R, ddtheta_2, ddl_3, ddl], dim=0).reshape(7, 1)

    nu_b_dot_meas = torch.cat([st[13:16].reshape(3, 1), st[16:19].reshape(3, 1)], dim=0)
    fp_code = st[38].item()
    return eta, v_b, w_b, nu_b, q_s, qd_s, qdd_s, nu_b_dot_meas, fp_code


def _compute_tau_wing_hydro_total(
    model: AWUGModelFromTex,
    nu_b: torch.Tensor,
    q_s: torch.Tensor,
    qd_s: torch.Tensor,
) -> torch.Tensor:
    """Compute total left+right wing hydrodynamic wrench in body coordinates."""
    theta_L, alpha_L, theta_R, alpha_R, _, _, _ = [q_s[i].reshape(1)[0] for i in range(7)]
    dtheta_L, dalpha_L, dtheta_R, dalpha_R, _, _, _ = [qd_s[i].reshape(1)[0] for i in range(7)]
    tau_wing = torch.zeros((6, 1), dtype=model.dtype, device=model.device)

    for wing in ("L", "R"):
        if wing == "L":
            theta_k, alpha_k, dalpha_k, dtheta_k, r_hk_B = theta_L, alpha_L, dalpha_L, dtheta_L, model.r_lh
        else:
            theta_k, alpha_k, dalpha_k, dtheta_k, r_hk_B = theta_R, alpha_R, dalpha_R, dtheta_R, model.r_rh

        R_K2B, J_k, _ = model._wing_terms(theta_k, alpha_k, dalpha_k)
        P_k = spatial_force_transform(R_K2B, r_hk_B)
        qd_k = torch.stack([dtheta_k, dalpha_k], dim=0).reshape(2, 1)
        nu_k = P_k.T @ nu_b + J_k @ qd_k
        tau_wing = tau_wing + P_k @ model._wing_hydrodynamic_wrench(nu_k, torch.abs(theta_k), wing)

    return tau_wing


def _ridge_lstsq(X: torch.Tensor, Y: torch.Tensor, ridge_lambda: float = 0.0) -> torch.Tensor:
    """Solve least squares with optional L2 regularization."""
    if ridge_lambda <= 0.0:
        return torch.linalg.lstsq(X, Y).solution
    n_cols = X.shape[1]
    sqrt_lam = math.sqrt(max(ridge_lambda, 0.0))
    eye = torch.eye(n_cols, dtype=X.dtype, device=X.device)
    zeros = torch.zeros((n_cols, Y.shape[1]), dtype=Y.dtype, device=Y.device)
    X_aug = torch.cat([X, sqrt_lam * eye], dim=0)
    Y_aug = torch.cat([Y, zeros], dim=0)
    return torch.linalg.lstsq(X_aug, Y_aug).solution


def _weighted_lstsq_axis_std(
    X: torch.Tensor,
    Y: torch.Tensor,
    block_dim: int,
    *,
    ridge_lambda: float = 0.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Solve LS after weighting rows by inverse target-axis standard deviation."""
    if X.shape[0] != Y.shape[0]:
        raise ValueError("X/Y row mismatch in weighted lsq")
    if X.shape[0] % block_dim != 0:
        raise ValueError(f"row count {X.shape[0]} not divisible by block_dim={block_dim}")

    y_blk = Y.reshape(-1, block_dim)
    std_axis = torch.std(y_blk, dim=0, unbiased=False).reshape(1, block_dim)
    w_axis = 1.0 / (std_axis + 1e-8)
    w_rows = w_axis.repeat(y_blk.shape[0], 1).reshape(-1, 1)

    Xw = X * w_rows
    Yw = Y * w_rows
    sol = _ridge_lstsq(Xw, Yw, ridge_lambda=ridge_lambda)
    return sol, w_axis


def _solve_lsq_with_options(
    X: torch.Tensor,
    Y: torch.Tensor,
    *,
    block_dim: int,
    weighted: bool,
    col_normalize: bool,
    ridge_lambda: float = 0.0,
) -> torch.Tensor:
    """Apply column normalization and/or axis weighting before LS solve."""
    if col_normalize:
        nrm = torch.linalg.vector_norm(X, dim=0).clamp_min(1e-12)
        Xn = X / nrm
        if weighted:
            theta_n, _ = _weighted_lstsq_axis_std(Xn, Y, block_dim=block_dim, ridge_lambda=ridge_lambda)
        else:
            theta_n = _ridge_lstsq(Xn, Y, ridge_lambda=ridge_lambda)
        return theta_n / nrm.reshape(-1, 1)

    if weighted:
        theta, _ = _weighted_lstsq_axis_std(X, Y, block_dim=block_dim, ridge_lambda=ridge_lambda)
    else:
        theta = _ridge_lstsq(X, Y, ridge_lambda=ridge_lambda)
    return theta


@torch.no_grad()
def build_body_ls_residual(
    model: AWUGModelFromTex,
    traj_list: List[torch.Tensor],
    *,
    verbose: bool = False,
    ls_method: str = "batch",
    rls_lambda: float = 1.0,
    rls_delta: float = 1e6,
    progress_every: int = 2000,
    weighted: bool = True,
    col_normalize: bool = False,
    ridge_lambda: float = 0.0,
    min_linear_speed: float = 0.0,
    min_angular_speed: float = 0.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Stage-A residual LS for fuselage hydrodynamic coefficients."""
    t0 = time.time()
    min_linear_speed = max(0.0, float(min_linear_speed))
    min_angular_speed = max(0.0, float(min_angular_speed))

    def keep_sample(v_b: torch.Tensor, w_b: torch.Tensor) -> bool:
        linear_speed = float(torch.linalg.norm(v_b).item())
        angular_speed = float(torch.linalg.norm(w_b).item())
        return (linear_speed >= min_linear_speed) or (angular_speed >= min_angular_speed)

    sample_count = 0
    used_count = 0
    skipped_low_excitation = 0

    if ls_method.lower() == "rls":
        theta3 = torch.zeros((10, 1), dtype=model.dtype, device=model.device)
        theta4 = torch.zeros((10, 1), dtype=model.dtype, device=model.device)
        P3 = torch.eye(10, dtype=model.dtype, device=model.device) * rls_delta
        P4 = torch.eye(10, dtype=model.dtype, device=model.device) * rls_delta

        for traj in traj_list:
            for i in range(traj.shape[0]):
                sample_count += 1
                if verbose and progress_every > 0 and sample_count % progress_every == 0:
                    print(f"[body RLS] samples={sample_count} used={used_count} elapsed={time.time()-t0:.1f}s")

                eta, v_b, w_b, nu_b, q_s, qd_s, qdd_s, nu_b_dot_meas, fp_code = _parse_state_for_residual(model, traj[i])
                if not keep_sample(v_b, w_b):
                    skipped_low_excitation += 1
                    continue
                used_count += 1

                M_q_s, C_nu, M_bs_qdd_s, b_s, g_q = model._accumulate_terms(nu_b, eta, q_s, qd_s, qdd_s)
                tau_ext_total = M_q_s @ nu_b_dot_meas + C_nu + M_bs_qdd_s + b_s + g_q
                tau_prop = torch.zeros((6, 1), dtype=model.dtype, device=model.device)
                tau_prop[0, 0] = _fp_code_to_F_p(fp_code, model)
                tau_wing_hydro = _compute_tau_wing_hydro_total(model, nu_b, q_s, qd_s)
                tau_body_hydro = tau_ext_total - tau_prop - tau_wing_hydro

                X3 = model._construct_X3_matrix(v_b, w_b, model.device, model.dtype)
                X4 = model._construct_X4_matrix(v_b, w_b, model.device, model.dtype)

                for j in range(3):
                    x = X3[j : j + 1, :]
                    y = tau_body_hydro[j : j + 1, :]
                    Px = P3 @ x.T
                    k = Px / (rls_lambda + x @ Px)
                    theta3 = theta3 + k @ (y - x @ theta3)
                    P3 = (P3 - k @ x @ P3) / rls_lambda

                for j in range(3):
                    x = X4[j : j + 1, :]
                    y = tau_body_hydro[3 + j : 4 + j, :]
                    Px = P4 @ x.T
                    k = Px / (rls_lambda + x @ Px)
                    theta4 = theta4 + k @ (y - x @ theta4)
                    P4 = (P4 - k @ x @ P4) / rls_lambda

        if used_count == 0:
            raise ValueError(
                "build_body_ls_residual: no valid samples after excitation filtering. "
                "Lower min_linear_speed/min_angular_speed."
            )
        theta3_eff = theta3.reshape(10, 1)
        theta4_eff = theta4.reshape(10, 1)
    else:
        X3_all: List[torch.Tensor] = []
        X4_all: List[torch.Tensor] = []
        Y3_all: List[torch.Tensor] = []
        Y4_all: List[torch.Tensor] = []

        for traj in traj_list:
            for i in range(traj.shape[0]):
                sample_count += 1
                if verbose and progress_every > 0 and sample_count % progress_every == 0:
                    print(f"[body LS] samples={sample_count} used={used_count} elapsed={time.time()-t0:.1f}s")

                eta, v_b, w_b, nu_b, q_s, qd_s, qdd_s, nu_b_dot_meas, fp_code = _parse_state_for_residual(model, traj[i])
                if not keep_sample(v_b, w_b):
                    skipped_low_excitation += 1
                    continue
                used_count += 1

                M_q_s, C_nu, M_bs_qdd_s, b_s, g_q = model._accumulate_terms(nu_b, eta, q_s, qd_s, qdd_s)
                tau_ext_total = M_q_s @ nu_b_dot_meas + C_nu + M_bs_qdd_s + b_s + g_q
                tau_prop = torch.zeros((6, 1), dtype=model.dtype, device=model.device)
                tau_prop[0, 0] = _fp_code_to_F_p(fp_code, model)
                tau_wing_hydro = _compute_tau_wing_hydro_total(model, nu_b, q_s, qd_s)
                tau_body_hydro = tau_ext_total - tau_prop - tau_wing_hydro

                X3_all.append(model._construct_X3_matrix(v_b, w_b, model.device, model.dtype))
                X4_all.append(model._construct_X4_matrix(v_b, w_b, model.device, model.dtype))
                Y3_all.append(tau_body_hydro[:3].reshape(3, 1))
                Y4_all.append(tau_body_hydro[3:].reshape(3, 1))

        if not X3_all or not X4_all:
            raise ValueError(
                "build_body_ls_residual: no valid LS rows after excitation filtering. "
                "Lower min_linear_speed/min_angular_speed."
            )
        X3 = torch.cat(X3_all, dim=0)
        X4 = torch.cat(X4_all, dim=0)
        Y3 = torch.cat(Y3_all, dim=0)
        Y4 = torch.cat(Y4_all, dim=0)

        theta3_eff = _solve_lsq_with_options(
            X3,
            Y3,
            block_dim=3,
            weighted=weighted,
            col_normalize=col_normalize,
            ridge_lambda=ridge_lambda,
        ).reshape(10, 1)
        theta4_eff = _solve_lsq_with_options(
            X4,
            Y4,
            block_dim=3,
            weighted=weighted,
            col_normalize=col_normalize,
            ridge_lambda=ridge_lambda,
        ).reshape(10, 1)

    theta3_full = torch.zeros_like(theta3_eff)
    theta4_full = torch.zeros_like(theta4_eff)

    theta3_full[0:2] = theta3_eff[0:2]
    theta3_full[2:4] = theta3_eff[6:8]
    theta3_full[4:6] = -theta3_eff[8:10]
    theta3_full[6:10] = theta3_eff[6:10]

    theta4_full[0:2] = theta4_eff[0:2]
    theta4_full[2:4] = theta4_eff[6:8]
    theta4_full[4:6] = -theta4_eff[8:10]
    theta4_full[6:10] = theta4_eff[6:10]

    if verbose:
        th3_norm = float(torch.linalg.norm(theta3_full).item())
        th4_norm = float(torch.linalg.norm(theta4_full).item())
        th3_max = float(torch.max(torch.abs(theta3_full)).item())
        th4_max = float(torch.max(torch.abs(theta4_full)).item())
        print(
            f"[body LS] done in {time.time()-t0:.1f}s, samples={sample_count}, "
            f"used={used_count}, filtered={skipped_low_excitation}, ridge={ridge_lambda:.3e}",
            flush=True,
        )
        print(
            f"[body LS] Theta_3 norm={th3_norm:.3e} max_abs={th3_max:.3e}; "
            f"Theta_4 norm={th4_norm:.3e} max_abs={th4_max:.3e}",
            flush=True,
        )
    return theta3_full, theta4_full


def _wing_linear_block(
    model: AWUGModelFromTex,
    nu_b: torch.Tensor,
    q_s: torch.Tensor,
    qd_s: torch.Tensor,
) -> torch.Tensor:
    """Build per-sample linear regressor A for `[Theta_5; Theta_6]`."""
    A_total = torch.zeros((6, WING_HYDRO_PARAM_DIM), dtype=model.dtype, device=model.device)

    theta_L, alpha_L, theta_R, alpha_R, _, _, _ = [q_s[i].reshape(1)[0] for i in range(7)]
    dtheta_L, dalpha_L, dtheta_R, dalpha_R, _, _, _ = [qd_s[i].reshape(1)[0] for i in range(7)]

    for wing in ("L", "R"):
        if wing == "L":
            theta_k, alpha_k, dtheta_k, dalpha_k, r_hk_B = theta_L, alpha_L, dtheta_L, dalpha_L, model.r_lh
        else:
            theta_k, alpha_k, dtheta_k, dalpha_k, r_hk_B = theta_R, alpha_R, dtheta_R, dalpha_R, model.r_rh

        R_K2B, J_k, _ = model._wing_terms(theta_k, alpha_k, dalpha_k)
        P_k = spatial_force_transform(R_K2B, r_hk_B)
        qd_k = torch.stack([dtheta_k, dalpha_k], dim=0).reshape(2, 1)
        nu_k = P_k.T @ nu_b + J_k @ qd_k

        v_k = nu_k[:3]
        w_k = nu_k[3:]
        theta_abs = torch.abs(theta_k)
        r_cp = model._r_cp(theta_abs, wing)
        v_aero = v_k + skew(w_k) @ r_cp
        V = torch.linalg.norm(v_aero) + 1e-10
        alpha = torch.atan2(v_aero[2, 0], v_aero[0, 0] + 1e-10)
        beta = torch.asin(torch.clamp(v_aero[1, 0] / V, -1.0, 1.0))

        _, s_k = model._S_eff(theta_abs)
        X5 = model._construct_X5_matrix(theta_abs, alpha, beta, V, s_k)
        X6_static_a, X6_rate_k = model._construct_X6_matrices(theta_abs, alpha, beta, V, s_k, w_k)
        R_wa = model._R_wa(alpha, beta)

        A5 = R_wa @ X5
        A6 = R_wa @ X6_static_a + X6_rate_k
        A5_m = skew(r_cp) @ A5

        A_k = torch.zeros((6, WING_HYDRO_PARAM_DIM), dtype=model.dtype, device=model.device)
        A_k[:3, :WING_FORCE_PARAM_DIM] = A5
        A_k[3:, :WING_FORCE_PARAM_DIM] = A5_m
        A_k[3:, WING_FORCE_PARAM_DIM:] = A6
        A_total = A_total + P_k @ A_k

    return A_total


@torch.no_grad()
def build_wing_ls_residual(
    model: AWUGModelFromTex,
    traj_list: List[torch.Tensor],
    *,
    verbose: bool = False,
    ls_method: str = "batch",
    rls_lambda: float = 1.0,
    rls_delta: float = 1e6,
    progress_every: int = 1000,
    weighted: bool = True,
    col_normalize: bool = False,
    ridge_lambda: float = 0.0,
    min_linear_speed: float = 0.0,
    min_angular_speed: float = 0.0,
    split_force_moment: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Stage-C residual LS for wing hydrodynamic coefficients."""
    t0 = time.time()
    min_linear_speed = max(0.0, float(min_linear_speed))
    min_angular_speed = max(0.0, float(min_angular_speed))

    def keep_sample(v_b: torch.Tensor, w_b: torch.Tensor) -> bool:
        linear_speed = float(torch.linalg.norm(v_b).item())
        angular_speed = float(torch.linalg.norm(w_b).item())
        return (linear_speed >= min_linear_speed) or (angular_speed >= min_angular_speed)

    if ls_method.lower() == "rls":
        P = torch.eye(WING_HYDRO_PARAM_DIM, dtype=model.dtype, device=model.device) * rls_delta
        theta = torch.zeros((WING_HYDRO_PARAM_DIM, 1), dtype=model.dtype, device=model.device)
        row_count = 0
        used_count = 0
        skipped_low_excitation = 0

        for traj in traj_list:
            for i in range(traj.shape[0]):
                row_count += 1
                if verbose and progress_every > 0 and row_count % progress_every == 0:
                    print(
                        f"[wing RLS] rows={row_count} used={used_count} "
                        f"filtered={skipped_low_excitation} elapsed={time.time()-t0:.1f}s"
                    )

                eta, v_b, w_b, nu_b, q_s, qd_s, qdd_s, nu_b_dot_meas, fp_code = _parse_state_for_residual(model, traj[i])
                if not keep_sample(v_b, w_b):
                    skipped_low_excitation += 1
                    continue
                used_count += 1
                M_q_s, C_nu, M_bs_qdd_s, b_s, g_q = model._accumulate_terms(nu_b, eta, q_s, qd_s, qdd_s)
                tau_ext_total = M_q_s @ nu_b_dot_meas + C_nu + M_bs_qdd_s + b_s + g_q
                tau_prop = torch.zeros((6, 1), dtype=model.dtype, device=model.device)
                tau_prop[0, 0] = _fp_code_to_F_p(fp_code, model)
                tau_body_hydro = model._fuselage_hydrodynamic_wrench(nu_b)
                tau_wing_res = tau_ext_total - tau_prop - tau_body_hydro

                A = _wing_linear_block(model, nu_b, q_s, qd_s)
                for j in range(6):
                    x = A[j : j + 1, :]
                    y = tau_wing_res[j : j + 1, :]
                    Px = P @ x.T
                    k = Px / (rls_lambda + x @ Px)
                    theta = theta + k @ (y - x @ theta)
                    P = (P - k @ x @ P) / rls_lambda

        if used_count == 0:
            raise ValueError(
                "build_wing_ls_residual: no valid samples after excitation filtering. "
                "Lower min_linear_speed/min_angular_speed."
            )
        theta_vec = theta.reshape(WING_HYDRO_PARAM_DIM, 1)
    else:
        A_all: List[torch.Tensor] = []
        Y_all: List[torch.Tensor] = []
        sample_count = 0
        used_count = 0
        skipped_low_excitation = 0

        for traj in traj_list:
            for i in range(traj.shape[0]):
                sample_count += 1
                if verbose and progress_every > 0 and sample_count % progress_every == 0:
                    print(
                        f"[wing LS] samples={sample_count} used={used_count} "
                        f"filtered={skipped_low_excitation} elapsed={time.time()-t0:.1f}s"
                    )

                eta, v_b, w_b, nu_b, q_s, qd_s, qdd_s, nu_b_dot_meas, fp_code = _parse_state_for_residual(model, traj[i])
                if not keep_sample(v_b, w_b):
                    skipped_low_excitation += 1
                    continue
                used_count += 1
                M_q_s, C_nu, M_bs_qdd_s, b_s, g_q = model._accumulate_terms(nu_b, eta, q_s, qd_s, qdd_s)
                tau_ext_total = M_q_s @ nu_b_dot_meas + C_nu + M_bs_qdd_s + b_s + g_q
                tau_prop = torch.zeros((6, 1), dtype=model.dtype, device=model.device)
                tau_prop[0, 0] = _fp_code_to_F_p(fp_code, model)
                tau_body_hydro = model._fuselage_hydrodynamic_wrench(nu_b)
                tau_wing_res = tau_ext_total - tau_prop - tau_body_hydro

                A_all.append(_wing_linear_block(model, nu_b, q_s, qd_s))
                Y_all.append(tau_wing_res)

        if not A_all:
            raise ValueError(
                "build_wing_ls_residual: no valid LS rows after excitation filtering. "
                "Lower min_linear_speed/min_angular_speed."
            )
        X = torch.cat(A_all, dim=0)
        Y = torch.cat(Y_all, dim=0)
        if split_force_moment:
            # Two-step Stage C:
            # 1) identify Theta_5 only from force rows, so drag/lift/side-force
            #    coefficients are not pulled by moment residuals or r_cp x F;
            # 2) subtract the identified force-induced moment and identify
            #    Theta_6 from moment rows.
            X_blk = X.reshape(-1, 6, WING_HYDRO_PARAM_DIM)
            Y_blk = Y.reshape(-1, 6, 1)

            X_force = X_blk[:, :3, :WING_FORCE_PARAM_DIM].reshape(-1, WING_FORCE_PARAM_DIM)
            Y_force = Y_blk[:, :3, :].reshape(-1, 1)
            theta_5 = _solve_lsq_with_options(
                X_force,
                Y_force,
                block_dim=3,
                weighted=weighted,
                col_normalize=col_normalize,
                ridge_lambda=ridge_lambda,
            ).reshape(WING_FORCE_PARAM_DIM, 1)

            X_moment_force = X_blk[:, 3:, :WING_FORCE_PARAM_DIM].reshape(-1, WING_FORCE_PARAM_DIM)
            X_moment = X_blk[:, 3:, WING_FORCE_PARAM_DIM:].reshape(-1, WING_MOMENT_PARAM_DIM)
            Y_moment = Y_blk[:, 3:, :].reshape(-1, 1)
            Y_moment_res = Y_moment - X_moment_force @ theta_5
            theta_6 = _solve_lsq_with_options(
                X_moment,
                Y_moment_res,
                block_dim=3,
                weighted=weighted,
                col_normalize=col_normalize,
                ridge_lambda=ridge_lambda,
            ).reshape(WING_MOMENT_PARAM_DIM, 1)
            theta_vec = torch.cat([theta_5, theta_6], dim=0)
        else:
            # Legacy joint LS over the full 6-D wing wrench.
            theta_vec = _solve_lsq_with_options(
                X,
                Y,
                block_dim=6,
                weighted=weighted,
                col_normalize=col_normalize,
                ridge_lambda=ridge_lambda,
            ).reshape(WING_HYDRO_PARAM_DIM, 1)

    theta_5 = theta_vec[:WING_FORCE_PARAM_DIM].reshape(WING_FORCE_PARAM_DIM, 1)
    theta_6 = theta_vec[WING_FORCE_PARAM_DIM:].reshape(WING_MOMENT_PARAM_DIM, 1)
    if verbose:
        mode = "split_force_then_moment" if (split_force_moment and ls_method.lower() != "rls") else "joint"
        print(
            f"[wing LS] done in {time.time()-t0:.1f}s, mode={mode}, ridge={ridge_lambda:.3e}, "
            f"min_linear_speed={min_linear_speed:.3e}, min_angular_speed={min_angular_speed:.3e}"
        )
    return theta_5, theta_6


@dataclass
class ConstrainedRaw:
    """Bounded trainable parameter wrapper.

    ``mode="sigmoid"`` keeps the legacy smooth box transform:
    ``value = lo + (hi-lo)*sigmoid(raw)``. ``mode="project"`` trains the
    raw tensor directly and clamps it back into the box after each optimizer
    step, avoiding sigmoid saturation near active bounds.
    """

    raw: nn.Parameter
    init: torch.Tensor
    ratio: float
    eps: float = 1e-8
    band: str = "relative"
    mode: str = "project"
    # abs_symmetric constrains to +/- ratio*abs(init), allowing sign changes.
    _lo: torch.Tensor = field(init=False, repr=False)
    _hi: torch.Tensor = field(init=False, repr=False)

    def __post_init__(self):
        with torch.no_grad():
            if self.band == "relative":
                lo = self.init * (1.0 - self.ratio)
                hi = self.init * (1.0 + self.ratio)
            elif self.band == "abs_scaled":
                mag = torch.maximum(
                    torch.abs(self.init),
                    torch.as_tensor(self.eps, dtype=self.init.dtype, device=self.init.device),
                )
                half = self.ratio * mag
                lo = self.init - half
                hi = self.init + half
            elif self.band == "abs_symmetric":
                mag = torch.maximum(
                    torch.abs(self.init),
                    torch.as_tensor(self.eps, dtype=self.init.dtype, device=self.init.device),
                )
                half = self.ratio * mag
                lo = -half
                hi = half
            else:
                raise ValueError(f"unknown ConstrainedRaw.band: {self.band!r}")
            if self.mode not in {"sigmoid", "project"}:
                raise ValueError(f"unknown ConstrainedRaw.mode: {self.mode!r}")
            lo_sorted = torch.minimum(lo, hi)
            hi_sorted = torch.maximum(lo, hi)
            object.__setattr__(self, "_lo", lo_sorted.clone())
            object.__setattr__(self, "_hi", hi_sorted.clone())
            if self.mode == "project":
                self.raw.copy_(torch.maximum(torch.minimum(self.init, self._hi), self._lo))
            else:
                self.raw.zero_()

    def value(self) -> torch.Tensor:
        if self.mode == "project":
            return self.raw
        s = torch.sigmoid(self.raw)
        return self._lo + (self._hi - self._lo) * s

    def project_(self) -> None:
        """Clamp the directly-trained raw tensor into the feasible box."""
        if self.mode != "project":
            return
        with torch.no_grad():
            self.raw.copy_(torch.maximum(torch.minimum(self.raw, self._hi), self._lo))


@dataclass
class PositiveDiagonalMatrixRaw:
    """Constrain a square matrix to a positive diagonal matrix."""

    raw: nn.Parameter
    init: torch.Tensor
    ratio: float
    eps: float = 1e-8
    band: str = "relative"
    mode: str = "project"
    min_positive: float = 1e-12
    _lo: torch.Tensor = field(init=False, repr=False)
    _hi: torch.Tensor = field(init=False, repr=False)
    _lo_diag: torch.Tensor = field(init=False, repr=False)
    _hi_diag: torch.Tensor = field(init=False, repr=False)

    def __post_init__(self):
        with torch.no_grad():
            if self.init.ndim != 2 or self.init.shape[0] != self.init.shape[1]:
                raise ValueError("PositiveDiagonalMatrixRaw requires a square matrix init")
            if self.raw.ndim != 1 or self.raw.numel() != self.init.shape[0]:
                raise ValueError("PositiveDiagonalMatrixRaw raw must be a diagonal vector")
            if self.band not in {"relative", "abs_scaled"}:
                raise ValueError(f"unknown PositiveDiagonalMatrixRaw.band: {self.band!r}")
            if self.mode not in {"sigmoid", "project"}:
                raise ValueError(f"unknown PositiveDiagonalMatrixRaw.mode: {self.mode!r}")

            floor = torch.as_tensor(self.min_positive, dtype=self.init.dtype, device=self.init.device)
            diag_center = torch.maximum(torch.diagonal(self.init).clone(), floor)
            if self.band == "relative":
                lo_diag = diag_center * (1.0 - self.ratio)
                hi_diag = diag_center * (1.0 + self.ratio)
            else:
                mag = torch.maximum(
                    torch.abs(diag_center),
                    torch.as_tensor(self.eps, dtype=self.init.dtype, device=self.init.device),
                )
                half = self.ratio * mag
                lo_diag = diag_center - half
                hi_diag = diag_center + half

            lo_sorted = torch.minimum(lo_diag, hi_diag)
            hi_sorted = torch.maximum(lo_diag, hi_diag)
            lo_pos = torch.maximum(lo_sorted, floor)
            hi_pos = torch.maximum(hi_sorted, lo_pos)
            init_diag = torch.maximum(torch.minimum(diag_center, hi_pos), lo_pos)

            object.__setattr__(self, "init", torch.diag(init_diag))
            object.__setattr__(self, "_lo_diag", lo_pos.clone())
            object.__setattr__(self, "_hi_diag", hi_pos.clone())
            object.__setattr__(self, "_lo", torch.diag(lo_pos).clone())
            object.__setattr__(self, "_hi", torch.diag(hi_pos).clone())
            if self.mode == "project":
                self.raw.copy_(init_diag)
            else:
                self.raw.zero_()

    def value(self) -> torch.Tensor:
        if self.mode == "project":
            diag = self.raw
        else:
            s = torch.sigmoid(self.raw)
            diag = self._lo_diag + (self._hi_diag - self._lo_diag) * s
        return torch.diag(diag)

    def project_(self) -> None:
        """Clamp the trained diagonal into the positive feasible box."""
        if self.mode != "project":
            return
        with torch.no_grad():
            self.raw.copy_(torch.maximum(torch.minimum(self.raw, self._hi_diag), self._lo_diag))


def _optimizer_lr_string(opt: torch.optim.Optimizer) -> str:
    """Format all optimizer parameter-group learning rates."""
    parts = []
    for i, pg in enumerate(opt.param_groups):
        name = pg.get("name")
        label = f"g{i}({name})" if name else f"g{i}"
        parts.append(f"{label}={float(pg['lr']):.6e}")
    return " ".join(parts)


def _optimizer_lr_dict(opt: torch.optim.Optimizer) -> Dict[str, float]:
    """Return current optimizer learning rates keyed by group name."""
    out: Dict[str, float] = {}
    for i, pg in enumerate(opt.param_groups):
        name = str(pg.get("name") or f"g{i}")
        out[name] = float(pg["lr"])
    return out


def build_named_parameter_lr_groups(
    group_param_names: Dict[str, Sequence[str]],
    group_lrs: Dict[str, float | None],
) -> Dict[str, Tuple[str, float]]:
    """Map trainable parameter names to optional optimizer LR groups."""
    named_lrs: Dict[str, Tuple[str, float]] = {}
    for group_name, value in group_lrs.items():
        if value is None:
            continue
        lr_value = float(value)
        if lr_value <= 0:
            raise ValueError(f"learning rate for {group_name} must be positive, got {lr_value}")
        for param_name in group_param_names.get(group_name, ()):
            if param_name in named_lrs:
                old_group = named_lrs[param_name][0]
                raise ValueError(
                    f"parameter {param_name!r} is assigned to both {old_group!r} and {group_name!r}"
                )
            named_lrs[param_name] = (group_name, lr_value)
    return named_lrs


def _fmt_tensor_value(x: torch.Tensor, *, precision: int = 6, max_elems: int = 60) -> str:
    """Format a tensor compactly for training logs."""
    arr = x.detach().cpu().numpy()
    flat = arr.reshape(-1)
    if flat.size > max_elems:
        shown = flat[:max_elems]
        suffix = f"...(+{flat.size - max_elems} elems)"
    else:
        shown = flat
        suffix = ""
    text = np.array2string(shown, precision=precision, separator=", ", max_line_width=200)
    return f"shape={tuple(arr.shape)} value={text}{suffix}"


def _print_all_params_online(
    model: AWUGModelFromTex,
    constrained_items: List[Tuple[str, ConstrainedRaw | None]],
    *,
    prefix: str = "         params(online)",
) -> None:
    """Print trainable and constrained parameters for debugging."""
    constrained_map = {n: c for n, c in constrained_items if c is not None}
    print(prefix + ":", flush=True)
    with torch.no_grad():
        for n, p in model.named_parameters():
            if n in constrained_map:
                v = constrained_map[n].value()
                print(f"           - {n} (constrained): {_fmt_tensor_value(v)}", flush=True)
            else:
                print(f"           - {n}: {_fmt_tensor_value(p)}", flush=True)


def _print_eval_param_snapshot(
    model: AWUGModelFromTex,
    constrained_items: List[Tuple[str, ConstrainedRaw | None]],
    *,
    names: Tuple[str, ...] | None = None,
    prefix: str = "[eval-pre]",
) -> None:
    constrained_map = {n: c for n, c in constrained_items if c is not None}
    if names is None:
        names = tuple(n for n, c in constrained_items if c is not None)
    if not names:
        print(f"{prefix} <no constrained parameters to print>", flush=True)
        return
    with torch.no_grad():
        for name in names:
            if name in constrained_map:
                value = constrained_map[name].value()
                print(f"{prefix} {name} (constrained) = {_fmt_tensor_value(value, precision=6)}", flush=True)
                continue
            if hasattr(model, name):
                value = getattr(model, name)
                print(f"{prefix} {name} = {_fmt_tensor_value(value, precision=6)}", flush=True)
                continue
            print(f"{prefix} {name} = <not found>", flush=True)


def _finite_loss(loss: torch.Tensor) -> bool:
    """Return True only when loss is finite."""
    return bool(torch.isfinite(loss).all().item())


def _build_functional_forward(
    model: AWUGModelFromTex,
    constrained_items: List[Tuple[str, ConstrainedRaw | None]],
):
    """Build a forward function that injects constrained parameter values."""
    def fw(st: torch.Tensor):
        param_map = {n: p for n, p in model.named_parameters()}
        for n, c in constrained_items:
            if c is not None:
                param_map[n] = c.value()
        merged = {**param_map, **{n: b for n, b in model.named_buffers()}}
        return functional_call(model, merged, (st,))

    return fw


def _project_constrained_items(constrained_items: List[Tuple[str, ConstrainedRaw | None]]) -> None:
    """Apply post-step projection for constraints that use projected GD."""
    for _, c in constrained_items:
        if c is not None:
            c.project_()


def _setup_optimizer(
    model: AWUGModelFromTex,
    constrained_items: List[Tuple[str, ConstrainedRaw | None]],
    lr: float,
    constrained_lrs: Dict[str, float] | None = None,
    named_parameter_lrs: Dict[str, Tuple[str, float]] | None = None,
) -> tuple[torch.optim.Optimizer, list[nn.Parameter], list[nn.Parameter]]:
    """Create an Adam optimizer over normal trainable params plus constrained raws."""
    constrained_lrs = {} if constrained_lrs is None else dict(constrained_lrs)
    named_parameter_lrs = {} if named_parameter_lrs is None else dict(named_parameter_lrs)
    constrained_names = {n for n, c in constrained_items if c is not None}
    trainable_params: list[nn.Parameter] = []
    base_trainable_params: list[nn.Parameter] = []
    named_group_params: Dict[str, list[nn.Parameter]] = {}
    named_group_lrs: Dict[str, float] = {}
    for n, p in model.named_parameters():
        if n in constrained_names:
            p.requires_grad_(False)
        elif p.requires_grad:
            trainable_params.append(p)
            group_spec = named_parameter_lrs.get(n)
            if group_spec is None:
                base_trainable_params.append(p)
            else:
                group_name, group_lr = group_spec
                group_lr = float(group_lr)
                if group_lr <= 0:
                    raise ValueError(f"learning rate for {group_name} must be positive, got {group_lr}")
                if group_name in named_group_lrs and abs(named_group_lrs[group_name] - group_lr) > 0.0:
                    raise ValueError(
                        f"optimizer group {group_name!r} has conflicting learning rates: "
                        f"{named_group_lrs[group_name]} and {group_lr}"
                    )
                named_group_lrs[group_name] = group_lr
                named_group_params.setdefault(group_name, []).append(p)
    param_groups = []
    if base_trainable_params:
        param_groups.append({"params": base_trainable_params, "lr": lr, "name": "model"})
    for group_name, params in named_group_params.items():
        param_groups.append({"params": params, "lr": named_group_lrs[group_name], "name": group_name})
    raw_params: list[nn.Parameter] = []
    for name, constraint in constrained_items:
        if constraint is None:
            continue
        group_lr = float(constrained_lrs.get(name, lr))
        if group_lr <= 0:
            raise ValueError(f"learning rate for {name} must be positive, got {group_lr}")
        raw_params.append(constraint.raw)
        param_groups.append({"params": [constraint.raw], "lr": group_lr, "name": name})
    if not param_groups:
        raise ValueError("train_gd: no trainable parameters or constrained raw parameters")
    opt = torch.optim.Adam(param_groups, lr=lr)
    _project_constrained_items(constrained_items)
    return opt, trainable_params, raw_params


def train_gd(
    model: AWUGModelFromTex,
    train_traj: List[torch.Tensor],
    test_traj: List[torch.Tensor],
    epochs: int,
    lr: float,
    grad_clip: float,
    constrain_M_ab: ConstrainedRaw | None = None,
    constrain_I_ab: ConstrainedRaw | None = None,
    constrain_M_al: ConstrainedRaw | None = None,
    constrain_I_al: ConstrainedRaw | None = None,
    constrain_Theta_3: ConstrainedRaw | None = None,
    constrain_Theta_4: ConstrainedRaw | None = None,
    constrain_Theta_5: ConstrainedRaw | None = None,
    constrain_Theta_6: ConstrainedRaw | None = None,
    use_traj_loss: bool = True,
    dt: float = 1.0 / 90.0,
    integrator: str = "rk4",
    batch_size_min: int = 5,
    batch_size_max: int = 30,
    batch_size_schedule: str = "linear",
    batch_skip: int = 1,
    norm_scale_1_13: torch.Tensor | None = None,
    channel_weight_1_13: torch.Tensor | None = None,
    train_progress_every: int = 30,
    shuffle_windows: bool = True,
    lr_plateau: bool = False,
    plateau_factor: float = 0.5,
    plateau_patience: int = 5,
    plateau_threshold: float = 1e-4,
    plateau_min_lr: Union[float, List[float]] = 0.0,
    plateau_cooldown: int = 0,
    max_windows_per_epoch: int = 0,
    eval_every: int = 1,
    eval_param_callback=None,
    best_checkpoint_path: str | None = None,
    restore_best_state: bool = False,
    best_checkpoint_metric: str = "test_loss",
    best_checkpoint_metric_fn: Optional[Callable[[AWUGModelFromTex], float]] = None,
    early_stop_patience: int = 5,
    early_stop_min_delta: float = 0.0,
    early_stop_min_epochs: int = 0,
    parameter_projection_callback=None,
    constrained_lrs: Dict[str, float] | None = None,
    named_parameter_lrs: Dict[str, Tuple[str, float]] | None = None,
) -> Dict[str, Any]:
    """Shared GD loop for Stage B/D/E."""
    constrained_items = [
        ("M_ab", constrain_M_ab),
        ("I_ab", constrain_I_ab),
        ("M_al", constrain_M_al),
        ("I_al", constrain_I_al),
        ("Theta_3", constrain_Theta_3),
        ("Theta_4", constrain_Theta_4),
        ("Theta_5", constrain_Theta_5),
        ("Theta_6", constrain_Theta_6),
    ]

    opt, trainable_params, raw_params = _setup_optimizer(
        model,
        constrained_items,
        lr,
        constrained_lrs=constrained_lrs,
        named_parameter_lrs=named_parameter_lrs,
    )
    if parameter_projection_callback is not None:
        parameter_projection_callback(model)
    fw = _build_functional_forward(model, constrained_items)
    step_fn = get_integrator_step(integrator)

    plateau_sched = None
    if lr_plateau:
        plateau_sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
            opt,
            mode="min",
            factor=plateau_factor,
            patience=plateau_patience,
            threshold=plateau_threshold,
            min_lr=plateau_min_lr,
            cooldown=plateau_cooldown,
        )

    print(f"[train_gd] init lr: {_optimizer_lr_string(opt)} integrator={integrator}", flush=True)
    if lr_plateau:
        print(
            f"[train_gd] ReduceLROnPlateau factor={plateau_factor} patience={plateau_patience} "
            f"threshold={plateau_threshold} min_lr={plateau_min_lr} cooldown={plateau_cooldown}",
            flush=True,
        )

    checkpoint_metric_name = str(best_checkpoint_metric or "test_loss")
    if checkpoint_metric_name not in {"test_loss", "rmse_norm"}:
        raise ValueError("best_checkpoint_metric must be one of: test_loss, rmse_norm")
    if checkpoint_metric_name != "test_loss" and best_checkpoint_metric_fn is None:
        raise ValueError("best_checkpoint_metric_fn is required when best_checkpoint_metric != 'test_loss'")
    early_stop_patience = max(0, int(early_stop_patience))
    early_stop_min_delta = max(0.0, float(early_stop_min_delta))
    early_stop_min_epochs = max(0, int(early_stop_min_epochs))
    if early_stop_patience > 0:
        print(
            f"[train_gd] EarlyStopping patience={early_stop_patience} "
            f"min_delta={early_stop_min_delta} min_epochs={early_stop_min_epochs} "
            f"metric={checkpoint_metric_name}",
            flush=True,
        )

    best_checkpoint_metric_value = float("inf")
    best_epoch = -1
    best_test_at_best_checkpoint = float("nan")
    best_state_dict = None
    best_constrained_raw = None
    hist_train: List[float] = []
    hist_test: List[float] = []
    hist_lr: List[float] = []
    hist_lr_groups: List[Dict[str, float]] = []
    t0_all = time.time()
    eval_stride = max(1, int(eval_every))
    no_improve_evals = 0
    stopped_early = False
    early_stop_reason = ""
    eval_only_mode = False

    for ep in range(epochs):
        if eval_only_mode:
            print("[train_gd] eval-only mode active; stopping after the first evaluation pass.", flush=True)
            break
        t0_ep = time.time()
        model.train()
        losses: List[float] = []
        n_steps = 0
        bs_used = None
        n_batches = 0

        if use_traj_loss:
            bs = get_current_batch_size(ep, epochs, batch_size_min, batch_size_max, batch_size_schedule)
            bs_used = bs
            batches = create_traj_batches(train_traj, bs, batch_skip)
            if shuffle_windows:
                np.random.shuffle(batches)
            n_batches_total = len(batches)
            if max_windows_per_epoch > 0 and n_batches_total > max_windows_per_epoch:
                batches = batches[:max_windows_per_epoch]
            n_batches = len(batches)
            cap_text = f"/{n_batches_total}" if n_batches_total != n_batches else ""

            print(
                f"[train_gd] epoch {ep + 1}/{epochs} lr={_optimizer_lr_string(opt)} "
                f"bs={bs} windows={n_batches}{cap_text}",
                flush=True,
            )

            for wi, (fi, bi) in enumerate(batches, start=1):
                bd = train_traj[fi][bi : bi + bs]
                if bd.shape[0] < bs:
                    continue
                state = bd[0].clone()
                preds: List[torch.Tensor] = []

                opt.zero_grad(set_to_none=True)
                for k in range(bs - 1):
                    state[13:] = bd[k, 13:]
                    state = step_fn(model, state, dt, forward_fn=fw)
                    preds.append(state.clone())

                if not preds:
                    continue
                pred = torch.stack(preds, dim=0)
                loss = traj_loss_mse(
                    pred,
                    bd[1:],
                    norm_scale_1_13=norm_scale_1_13,
                    channel_weight_1_13=channel_weight_1_13,
                )
                if not _finite_loss(loss):
                    continue

                if not loss.requires_grad:
                    if not eval_only_mode:
                        print(
                            "[train_gd] warning: loss has no grad path for this variant; "
                            "switching to eval-only mode.",
                            flush=True,
                        )
                    eval_only_mode = True
                    val = float(loss.item())
                    losses.append(val)
                    n_steps += 1
                    break

                loss.backward()
                if grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(trainable_params + raw_params, grad_clip)
                opt.step()
                _project_constrained_items(constrained_items)
                if parameter_projection_callback is not None:
                    parameter_projection_callback(model)

                val = float(loss.item())
                losses.append(val)
                n_steps += 1

                if train_progress_every > 0 and (wi == 1 or wi % train_progress_every == 0 or wi == n_batches):
                    print(
                        f"  ... window {wi}/{n_batches} loss={val:.6e} lr={_optimizer_lr_string(opt)} "
                        f"elapsed={time.time()-t0_ep:.1f}s",
                        flush=True,
                    )
        else:
            print(
                f"[train_gd] epoch {ep + 1}/{epochs} lr={_optimizer_lr_string(opt)} pointwise mode",
                flush=True,
            )
            for traj in train_traj:
                for i in range(traj.shape[0]):
                    st = traj[i]
                    out = fw(st)
                    target = torch.stack([st[13], st[14], st[15], st[16], st[17], st[18]], dim=0).reshape(6, 1)
                    loss = torch.mean((out["nu_b_dot"] - target) ** 2)
                    if not _finite_loss(loss):
                        continue

                    opt.zero_grad(set_to_none=True)
                    loss.backward()
                    if grad_clip > 0:
                        torch.nn.utils.clip_grad_norm_(trainable_params + raw_params, grad_clip)
                    opt.step()
                    _project_constrained_items(constrained_items)
                    if parameter_projection_callback is not None:
                        parameter_projection_callback(model)

                    losses.append(float(loss.item()))
                    n_steps += 1

        train_loss = float(np.mean(losses)) if losses else 0.0
        train_min = float(min(losses)) if losses else 0.0
        train_max = float(max(losses)) if losses else 0.0
        ep_train_sec = time.time() - t0_ep

        should_eval = (ep % eval_stride == 0) or (ep == epochs - 1)
        if should_eval:
            print(f"[train_gd] eval-pre epoch {ep + 1}/{epochs} parameters:", flush=True)
            _print_eval_param_snapshot(model, constrained_items)
            if eval_param_callback is not None:
                eval_param_callback(model)
            model.eval()
            test_loss = rollout_loss_value(
                model,
                test_traj,
                dt=dt,
                batch_size=batch_size_max,
                batch_skip=batch_skip,
                norm_scale_1_13=norm_scale_1_13,
                channel_weight_1_13=channel_weight_1_13,
                forward_fn=fw,
                integrator=integrator,
            )
            ep_eval_sec = time.time() - t0_ep - ep_train_sec
            test_loss_f = float(test_loss)
            if checkpoint_metric_name == "test_loss":
                checkpoint_metric_f = test_loss_f
            else:
                checkpoint_metric_f = float(best_checkpoint_metric_fn(model))
        else:
            test_loss_f = float("nan")
            checkpoint_metric_f = float("nan")
            ep_eval_sec = 0.0

        lr_before_groups = _optimizer_lr_dict(opt)
        if plateau_sched is not None and should_eval and math.isfinite(test_loss_f):
            plateau_sched.step(test_loss_f)
        lr_after_groups = _optimizer_lr_dict(opt)
        lr_after = float(opt.param_groups[0]["lr"])
        lr_reduced = any(
            lr_after_groups.get(name, before) < before - 1e-18
            for name, before in lr_before_groups.items()
        )
        if plateau_sched is not None and lr_reduced:
            print(
                f"[train_gd] lr reduced: "
                f"{' '.join(f'{k}={v:.3e}' for k, v in lr_before_groups.items())} -> "
                f"{' '.join(f'{k}={v:.3e}' for k, v in lr_after_groups.items())}",
                flush=True,
            )

        hist_lr.append(lr_after)
        hist_lr_groups.append(lr_after_groups)
        improved = (
            math.isfinite(checkpoint_metric_f)
            and checkpoint_metric_f < best_checkpoint_metric_value - early_stop_min_delta
        )
        if improved:
            best_checkpoint_metric_value = checkpoint_metric_f
            best_epoch = ep + 1
            best_test_at_best_checkpoint = test_loss_f
            no_improve_evals = 0
            best_state_dict = {
                key: value.detach().cpu().clone()
                for key, value in deepcopy(model.state_dict()).items()
            }
            best_constrained_raw = {
                name: c.raw.detach().cpu().clone()
                for name, c in constrained_items
                if c is not None
            }
            if best_checkpoint_path:
                ckpt_dir = os.path.dirname(best_checkpoint_path)
                if ckpt_dir:
                    os.makedirs(ckpt_dir, exist_ok=True)
                torch.save(
                    {
                        "epoch": best_epoch,
                        "test_loss": test_loss_f,
                        "best_checkpoint_metric_name": checkpoint_metric_name,
                        "best_checkpoint_metric_value": best_checkpoint_metric_value,
                        "train_loss": train_loss,
                        "lr": lr_after,
                        "model_state_dict": best_state_dict,
                        "constrained_raw": best_constrained_raw,
                    },
                    best_checkpoint_path,
                )
                print(
                    f"[train_gd] saved best checkpoint: epoch={best_epoch} "
                    f"test={test_loss_f:.6e} {checkpoint_metric_name}={best_checkpoint_metric_value:.6e} "
                    f"path={best_checkpoint_path}",
                    flush=True,
                )
        elif should_eval and math.isfinite(checkpoint_metric_f):
            no_improve_evals += 1
        hist_train.append(train_loss)
        hist_test.append(test_loss_f)

        extra = f"bs={bs_used} windows={n_batches}" if use_traj_loss else "pointwise"
        eval_text = f"{test_loss_f:.6e}" if math.isfinite(test_loss_f) else "skipped"
        metric_text = (
            f"{checkpoint_metric_name}={checkpoint_metric_f:.6e}"
            if math.isfinite(checkpoint_metric_f)
            else f"{checkpoint_metric_name}=skipped"
        )
        print(
            f"Epoch [{ep+1}/{epochs}] {extra} steps={n_steps} "
            f"train={train_loss:.6e} train[min,max]=({train_min:.6e},{train_max:.6e}) "
            f"test={eval_text} {metric_text} best_{checkpoint_metric_name}={best_checkpoint_metric_value:.6e} "
            f"lr={_optimizer_lr_string(opt)} "
            f"time={time.time()-t0_ep:.2f}s(train={ep_train_sec:.2f}s eval={ep_eval_sec:.2f}s)",
            flush=True,
        )
        if eval_only_mode:
            print("[train_gd] eval-only mode completed after one pass.", flush=True)
            break
        if (
            early_stop_patience > 0
            and should_eval
            and math.isfinite(checkpoint_metric_f)
            and ep + 1 >= early_stop_min_epochs
            and no_improve_evals >= early_stop_patience
        ):
            stopped_early = True
            early_stop_reason = (
                f"no improvement in {no_improve_evals} evaluated epochs "
                f"(best_epoch={best_epoch}, best_{checkpoint_metric_name}={best_checkpoint_metric_value:.6e})"
            )
            print(f"[train_gd] early stopping: {early_stop_reason}", flush=True)
            break

    total_sec = time.time() - t0_all
    restored_best = False
    if restore_best_state and best_state_dict is not None:
        model.load_state_dict(best_state_dict)
        if best_constrained_raw is not None:
            for name, c in constrained_items:
                if c is not None and name in best_constrained_raw:
                    c.raw.data.copy_(best_constrained_raw[name].to(device=c.raw.device, dtype=c.raw.dtype))
                    c.project_()
        if parameter_projection_callback is not None:
            parameter_projection_callback(model)
        restored_best = True
        print(
            f"[train_gd] restored best checkpoint from epoch={best_epoch} "
            f"test={best_test_at_best_checkpoint:.6e} {checkpoint_metric_name}={best_checkpoint_metric_value:.6e}",
            flush=True,
        )
    print(
        f"[train_gd done] total={total_sec:.2f}s ({total_sec/60.0:.2f} min) "
        f"best_{checkpoint_metric_name}={best_checkpoint_metric_value:.6e} final_lr={_optimizer_lr_string(opt)}",
        flush=True,
    )
    return {
        "best_test_loss": best_test_at_best_checkpoint,
        "best_checkpoint_metric_name": checkpoint_metric_name,
        "best_checkpoint_metric_value": best_checkpoint_metric_value,
        "best_epoch": best_epoch,
        "best_checkpoint_path": best_checkpoint_path,
        "restored_best_state": restored_best,
        "stopped_early": stopped_early,
        "early_stop_reason": early_stop_reason,
        "early_stop_patience": early_stop_patience,
        "early_stop_min_delta": early_stop_min_delta,
        "early_stop_min_epochs": early_stop_min_epochs,
        "train_loss_hist": hist_train,
        "test_loss_hist": hist_test,
        "lr_hist": hist_lr,
        "lr_group_hist": hist_lr_groups,
    }


def _ensure_matplotlib():
    try:
        import matplotlib.pyplot as plt  # noqa: F401
    except Exception as e:
        raise RuntimeError("matplotlib is required for plotting utilities") from e


def plot_convergence(train_hist: List[float], test_hist: List[float], out_dir: str, fname: str = "convergence.png"):
    _ensure_matplotlib()
    import matplotlib.pyplot as plt

    os.makedirs(out_dir, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 4))
    if train_hist:
        ax.semilogy(train_hist, label="train")
    if test_hist:
        ax.semilogy(test_hist, label="test")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("GD Convergence")
    ax.grid(True)
    ax.legend()
    path = os.path.join(out_dir, fname)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[saved] {path}")


def simulate_window(
    model: AWUGModelFromTex,
    traj_list: List[torch.Tensor],
    dt: float,
    batch_size: int,
    batch_skip: int = 1,
    window_index: int = -1,
    forward_fn=None,
    integrator: str = "rk4",
) -> Tuple[np.ndarray | None, np.ndarray | None]:
    batches = create_traj_batches(traj_list, batch_size, batch_skip=batch_skip)
    if not batches:
        return None, None
    if window_index < 0:
        window_index = len(batches) + window_index
    window_index = max(0, min(int(window_index), len(batches) - 1))
    fi, bi = batches[window_index]
    bd = traj_list[fi][bi : bi + batch_size]
    state = bd[0].clone()
    preds: List[torch.Tensor] = []
    step_fn = get_integrator_step(integrator)
    with torch.no_grad():
        for k in range(batch_size - 1):
            state[13:] = bd[k, 13:]
            state = step_fn(model, state, dt, forward_fn=forward_fn)
            preds.append(state.clone())
    meas = bd[1:].detach().cpu().numpy()
    pred = torch.stack(preds, dim=0).detach().cpu().numpy() if preds else None
    return meas, pred


def plot_comparison_window(
    traj_list: List[torch.Tensor],
    model_init: AWUGModelFromTex,
    model_tuned: AWUGModelFromTex,
    dt: float,
    batch_size: int,
    out_dir: str,
    fname: str = "comparison.png",
    batch_skip: int = 1,
    window_index: int = -1,
    integrator: str = "rk4",
):
    _ensure_matplotlib()
    import matplotlib.pyplot as plt

    os.makedirs(out_dir, exist_ok=True)
    true_np, init_np = simulate_window(
        model_init,
        traj_list,
        dt,
        batch_size,
        batch_skip=batch_skip,
        window_index=window_index,
        integrator=integrator,
    )
    _, tuned_np = simulate_window(
        model_tuned,
        traj_list,
        dt,
        batch_size,
        batch_skip=batch_skip,
        window_index=window_index,
        integrator=integrator,
    )
    if true_np is None or init_np is None or tuned_np is None:
        print("[warn] no available window for comparison plot")
        return

    labels_idx = [
        (1, "p_x"),
        (2, "p_y"),
        (3, "p_z"),
        (4, "e_phi_deg"),
        (5, "e_theta_deg"),
        (6, "e_psi_deg"),
        (7, "v_x"),
        (8, "v_y"),
        (9, "v_z"),
        (10, "w_x"),
        (11, "w_y"),
        (12, "w_z"),
    ]
    fig, axes = plt.subplots(4, 3, figsize=(15, 12))
    for ax, (idx, label) in zip(axes.flat, labels_idx):
        tr = true_np[:, idx].copy()
        ip = init_np[:, idx].copy()
        gd = tuned_np[:, idx].copy()
        if label.startswith("e_"):
            tr *= 180.0 / np.pi
            ip *= 180.0 / np.pi
            gd *= 180.0 / np.pi
        ax.plot(tr, label="meas", lw=2)
        ax.plot(ip, label="init", lw=1.5, ls="--", alpha=0.7)
        ax.plot(gd, label="tuned", lw=1.5, ls="-.")
        ax.set_title(label)
        ax.grid(True)
        ax.legend(fontsize=7)
    plt.tight_layout()
    path = os.path.join(out_dir, fname)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[saved] {path}")


def plot_test_rollout_windows(
    model: AWUGModelFromTex,
    traj_list: List[torch.Tensor],
    dt: float,
    batch_size: int,
    out_dir: str,
    *,
    prefix: str = "test_rollout",
    batch_skip: int = 1,
    num_windows: int = 3,
    forward_fn=None,
    integrator: str = "rk4",
) -> List[str]:
    """Plot measured vs predicted rollout windows from the test set."""
    _ensure_matplotlib()
    import matplotlib.pyplot as plt

    os.makedirs(out_dir, exist_ok=True)
    batches = create_traj_batches(traj_list, batch_size=batch_size, batch_skip=batch_skip)
    if not batches:
        print("[warn] no available test rollout window for plotting", flush=True)
        return []

    count = max(1, int(num_windows))
    if len(batches) <= count:
        window_indices = list(range(len(batches)))
    else:
        window_indices = np.linspace(0, len(batches) - 1, count, dtype=int).tolist()

    labels_idx = [
        (1, "p_x"),
        (2, "p_y"),
        (3, "p_z"),
        (4, "e_phi_deg"),
        (5, "e_theta_deg"),
        (6, "e_psi_deg"),
        (7, "v_x"),
        (8, "v_y"),
        (9, "v_z"),
        (10, "w_x"),
        (11, "w_y"),
        (12, "w_z"),
    ]
    saved_paths: List[str] = []
    summary_rows: List[Dict[str, Union[int, float, str]]] = []

    for out_idx, window_index in enumerate(window_indices, start=1):
        true_np, pred_np = simulate_window(
            model,
            traj_list,
            dt,
            batch_size,
            batch_skip=batch_skip,
            window_index=int(window_index),
            forward_fn=forward_fn,
            integrator=integrator,
        )
        if true_np is None or pred_np is None:
            continue

        true_plot = true_np.copy()
        pred_plot = pred_np.copy()
        true_plot[:, 4:7] *= 180.0 / math.pi
        pred_plot[:, 4:7] *= 180.0 / math.pi

        err = pred_np[:, 1:13] - true_np[:, 1:13]
        rmse_total = float(np.sqrt(np.mean(np.square(err)) + 1e-12))
        fi, bi = batches[int(window_index)]
        summary_rows.append(
            {
                "plot_index": int(out_idx),
                "window_index": int(window_index),
                "file_idx": int(fi),
                "begin_idx": int(bi),
                "batch_size": int(batch_size),
                "rmse_total": rmse_total,
            }
        )

        fig, axes = plt.subplots(4, 3, figsize=(15, 12))
        t = np.arange(true_plot.shape[0]) * dt
        for ax, (col, label) in zip(axes.reshape(-1), labels_idx):
            tr = true_plot[:, col]
            pr = pred_plot[:, col]
            ax.plot(t, tr, label="meas", lw=2)
            ax.plot(t, pr, label="pred", lw=1.5, ls="--")
            ax.set_title(label)
            ax.grid(True, alpha=0.3)
        axes[0, 0].legend()
        fig.suptitle(
            f"{prefix} window={window_index} file_idx={fi} begin={bi} rmse={rmse_total:.3e}"
        )
        plt.tight_layout()
        path = os.path.join(out_dir, f"{prefix}_window_{out_idx:02d}_idx_{int(window_index):05d}.png")
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        saved_paths.append(path)
        print(f"[saved] {path}", flush=True)

    if summary_rows:
        csv_path = os.path.join(out_dir, f"{prefix}_summary.csv")
        json_path = os.path.join(out_dir, f"{prefix}_summary.json")
        pd.DataFrame(summary_rows).to_csv(csv_path, index=False)
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(summary_rows, f, ensure_ascii=False, indent=2)
        print(f"[saved] {csv_path}", flush=True)
        print(f"[saved] {json_path}", flush=True)
        saved_paths.extend([csv_path, json_path])
    return saved_paths


@torch.no_grad()
def eval_windows_rmse(
    model: AWUGModelFromTex,
    traj_list: List[torch.Tensor],
    dt: float,
    batch_size: int,
    batch_skip: int = 1,
    norm_scale_1_13: torch.Tensor | None = None,
    integrator: str = "rk4",
) -> pd.DataFrame:
    rows = []
    batches = create_traj_batches(traj_list, batch_size=batch_size, batch_skip=batch_skip)
    step_fn = get_integrator_step(integrator)
    for fi, bi in batches:
        bd = traj_list[fi][bi : bi + batch_size]
        if bd.shape[0] < batch_size:
            continue
        state = bd[0].clone()
        preds: List[torch.Tensor] = []
        for k in range(batch_size - 1):
            state[13:] = bd[k, 13:]
            state = step_fn(model, state, dt)
            preds.append(state.clone())
        if not preds:
            continue
        pred = torch.stack(preds, dim=0)
        err = pred[:, 1:13] - bd[1:, 1:13]
        if norm_scale_1_13 is not None:
            err = err / (norm_scale_1_13 + 1e-12)

        mse_dim = torch.mean(err**2, dim=0)
        rmse_dim = torch.sqrt(mse_dim + 1e-12).detach().cpu().numpy()
        rmse_total = float(torch.sqrt(torch.mean(err**2) + 1e-12).item())
        rows.append(
            {
                "file_idx": int(fi),
                "begin_idx": int(bi),
                "rmse_total": rmse_total,
                **{f"rmse_dim_{j}": float(rmse_dim[j]) for j in range(12)},
            }
        )
    return pd.DataFrame(rows)


def save_window_eval(df: pd.DataFrame, out_dir: str, prefix: str):
    os.makedirs(out_dir, exist_ok=True)
    csv_path = os.path.join(out_dir, f"{prefix}_windows.csv")
    json_path = os.path.join(out_dir, f"{prefix}_windows.json")
    df.to_csv(csv_path, index=False)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(df.to_dict(orient="records"), f, ensure_ascii=False, indent=2)
    print(f"[saved] {csv_path}")
    print(f"[saved] {json_path}")


def plot_window_rmse_boxplot(df: pd.DataFrame, out_dir: str, fname: str = "rmse_boxplot.png"):
    _ensure_matplotlib()
    import matplotlib.pyplot as plt

    os.makedirs(out_dir, exist_ok=True)
    cols = ["rmse_total"] + [f"rmse_dim_{j}" for j in range(12) if f"rmse_dim_{j}" in df.columns]
    data = [df[c].dropna().to_numpy() for c in cols]

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.boxplot(data, showfliers=False)
    ax.set_xticks(range(1, len(cols) + 1))
    ax.set_xticklabels(cols, rotation=45, ha="right")
    ax.set_ylabel("RMSE")
    ax.set_title("Window-wise RMSE distribution")
    ax.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    path = os.path.join(out_dir, fname)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[saved] {path}")


# =============================================================================
# 5. Stage-shared loading, model construction, constraints and GD wrappers
# =============================================================================
@dataclass
class StageDataBundle:
    """Container passed from data loading into Stage A-E training/evaluation."""
    train_files: list[str]
    test_files: list[str]
    train_traj: list[torch.Tensor]
    test_traj: list[torch.Tensor]
    dt: float
    norm_scale: Optional[torch.Tensor]
    loss_weight: Optional[torch.Tensor]


def start_stage(stage_name: str, out_dir: str, argv: Sequence[str], *, base_dir: str) -> tuple[str, str]:
    """Create a stage-specific log directory and tee stdout/stderr into it."""
    root = os.path.join(base_dir, out_dir)
    return open_stage_log(root, stage_name, list(argv))


def _resolve_optional_path(path: Optional[str], *, base_dir: str) -> Optional[str]:
    """Resolve optional CLI paths relative to code_folder."""
    if not path:
        return None
    if os.path.isabs(path):
        return path
    return os.path.join(base_dir, path)


def load_stage_data(
    *,
    train_files_arg: str,
    test_files_arg: str,
    device: str,
    sample_step: int,
    dt_base: float,
    norm_mode: str = "none",
    norm_stats_json: Optional[str] = None,
    loss_weight_mode: str = "none",
    variance_weights_json: Optional[str] = None,
    task_weights_spec: Optional[str] = None,
    base_dir: str,
    include_test: bool = True,
) -> StageDataBundle:
    """Parse file lists, build trajectory tensors, and load optional normalization."""
    train_files = parse_files(train_files_arg)
    test_files = parse_files(test_files_arg) if test_files_arg else []
    train_traj = build_traj_data(train_files, device=device, sample_step=sample_step)
    test_traj = []
    if include_test and test_files:
        test_traj = build_traj_data(test_files, device=device, sample_step=sample_step)

    norm_scale = None
    if norm_mode != "none":
        stats_path = _resolve_optional_path(norm_stats_json, base_dir=base_dir)
        if not stats_path:
            raise ValueError("norm_mode requires norm_stats_json")
        norm_scale = load_norm_scale(stats_path, mode=norm_mode, device=device)

    variance_path = _resolve_optional_path(variance_weights_json, base_dir=base_dir)
    loss_weight = load_loss_channel_weight(
        mode=loss_weight_mode,
        norm_mode=norm_mode,
        device=device,
        variance_weights_json=variance_path,
        task_weights_spec=task_weights_spec,
    )

    dt = dt_base * max(1, sample_step)
    return StageDataBundle(
        train_files=train_files,
        test_files=test_files,
        train_traj=train_traj,
        test_traj=test_traj,
        dt=dt,
        norm_scale=norm_scale,
        loss_weight=loss_weight,
    )


def clone_base_params() -> Dict:
    """Return an independent copy of the nominal AWUG parameter dictionary."""
    return copy.deepcopy(mwauv_params)


def apply_body_state(params: Dict, stage_json: Dict) -> Dict:
    """Merge Stage A/B body parameters into an AWUG parameter dict."""
    if "body_added_mass_inertia" in stage_json:
        params["M_ab"] = stage_json["body_added_mass_inertia"]["M_ab"]
        params["I_ab"] = stage_json["body_added_mass_inertia"]["I_ab"]
    if "body_hydro_params" in stage_json:
        params["Theta_3"] = stage_json["body_hydro_params"]["Theta_3"]
        params["Theta_4"] = stage_json["body_hydro_params"]["Theta_4"]
    return params


def apply_wing_ls_state(params: Dict, stage_json: Dict) -> Dict:
    """Merge Stage C/D wing hydrodynamic parameters into an AWUG parameter dict."""
    if "wing_hydro_params" in stage_json:
        params["Theta_5"] = stage_json["wing_hydro_params"]["Theta_5"]
        params["Theta_6"] = stage_json["wing_hydro_params"]["Theta_6"]
    return params


def apply_wing_gd_state(params: Dict, stage_json: Dict) -> Dict:
    """Merge Stage D wing inertia and hydrodynamic parameters."""
    if "wing_added_mass_inertia" in stage_json:
        params["M_al"] = stage_json["wing_added_mass_inertia"]["M_al"]
        params["M_ar"] = stage_json["wing_added_mass_inertia"]["M_al"]
        params["I_al"] = stage_json["wing_added_mass_inertia"]["I_al"]
        params["I_ar"] = stage_json["wing_added_mass_inertia"]["I_al"]
    return apply_wing_ls_state(params, stage_json)


def apply_gate_state(params: Dict, stage_json: Dict) -> Dict:
    """Merge Stage E gate parameters into an AWUG parameter dict."""
    gates = stage_json.get("wing_hydro_gates", {})
    if not gates:
        return params

    for key in (
        "k_XA_x",
        "k_XA_y",
        "k_XA_z",
        "k_XA_x2",
        "k_XA_y2",
        "k_XA_z2",
        "k_KA",
        "k_MA",
        "k_NA",
        "k_KA2",
        "k_MA2",
        "k_NA2",
    ):
        if key in gates:
            params[key] = gates[key]

    if "k_g2_X5" in gates:
        params["k_g2_X5"] = gates["k_g2_X5"]
    if "k_g2_X6" in gates:
        params["k_g2_X6"] = gates["k_g2_X6"]
    return params


def load_stage_json(path: str) -> Dict:
    """Load a stage JSON result file."""
    return load_json(path)


def build_model(
    *,
    device: str,
    params: Optional[Dict] = None,
    train_body_added_mass_inertia: bool = False,
    train_body_hydro_params: bool = False,
    train_wing_added_mass_inertia: bool = False,
    train_wing_hydro_params: bool = False,
    train_wing_added_mass_inertia_gates: bool = False,
    train_wing_hydro_gates: bool = False,
    freeze_k2: bool = False,
    enable_sweep_gates: bool = True,
    enable_pressure_center_migration: bool = True,
    enable_added_mass_update: bool = True,
    freeze_sweep_geometry: bool = False,
    freeze_added_mass_scaling: bool = False,
    decouple_gate_eta_from_geometry_freeze: bool = False,
) -> AWUGModelFromTex:
    """Instantiate AWUGModelFromTex with a cloned parameter dictionary."""
    model_params = clone_base_params() if params is None else copy.deepcopy(params)
    return AWUGModelFromTex(
        params=model_params,
        device=device,
        train_body_added_mass_inertia=train_body_added_mass_inertia,
        train_body_hydro_params=train_body_hydro_params,
        train_wing_added_mass_inertia=train_wing_added_mass_inertia,
        train_wing_hydro_params=train_wing_hydro_params,
        train_wing_added_mass_inertia_gates=train_wing_added_mass_inertia_gates,
        train_wing_hydro_gates=train_wing_hydro_gates,
        freeze_k2=freeze_k2,
        enable_sweep_gates=enable_sweep_gates,
        enable_pressure_center_migration=enable_pressure_center_migration,
        enable_added_mass_update=enable_added_mass_update,
        freeze_sweep_geometry=freeze_sweep_geometry,
        freeze_added_mass_scaling=freeze_added_mass_scaling,
        decouple_gate_eta_from_geometry_freeze=decouple_gate_eta_from_geometry_freeze,
    )


def make_constraint(
    parameter: torch.Tensor,
    *,
    ratio: float,
    band: str,
    eps: float,
    mode: str = "project",
) -> ConstrainedRaw:
    """Wrap a model parameter in a bounded raw-parameter transform."""
    return ConstrainedRaw(
        nn.Parameter(torch.zeros_like(parameter)),
        parameter.detach().clone(),
        ratio,
        eps=eps,
        band=band,
        mode=mode,
    )


def make_positive_diagonal_matrix_constraint(
    parameter: torch.Tensor,
    *,
    ratio: float,
    band: str,
    eps: float,
    mode: str = "project",
) -> PositiveDiagonalMatrixRaw:
    """Wrap a square matrix as a positive diagonal-only trainable constraint."""
    return PositiveDiagonalMatrixRaw(
        nn.Parameter(torch.zeros(parameter.shape[0], dtype=parameter.dtype, device=parameter.device)),
        parameter.detach().clone(),
        ratio,
        eps=eps,
        band=band,
        mode=mode,
    )


def pretty_vector(tensor: torch.Tensor) -> list[float]:
    """Convert a tensor to a flat Python list for readable logs."""
    return tensor.detach().cpu().reshape(-1).tolist()

@dataclass(frozen=True)
class GdDefaults:
    """Default GD hyperparameters supplied by each stage before CLI overrides."""
    epochs: int
    lr: float
    grad_clip: float
    dt_base: float
    batch_size_min: int
    batch_size_max: int
    batch_size_schedule: str = "linear"
    batch_skip: int = 1
    norm_mode: str = "none"
    norm_stats_json: str | None = None
    loss_weight_mode: str = "none"
    variance_weights_json: str | None = None
    task_weights_spec: str | None = None
    train_progress_every: int = 30
    plateau_factor: float = 0.5
    plateau_patience: int = 5
    plateau_threshold: float = 1e-4
    plateau_min_lr: float = 1e-8
    plateau_cooldown: int = 0
    max_windows_per_epoch: int = 0
    eval_every: int = 1
    integrator: str = "rk4"
    early_stop_patience: int = 5
    early_stop_min_delta: float = 0.0
    early_stop_min_epochs: int = 0


def add_gd_arguments(parser, *, defaults: GdDefaults) -> None:
    """Register shared gradient-descent arguments used by stages B/D/E."""
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--sample-step", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=defaults.epochs)
    parser.add_argument("--lr", type=float, default=defaults.lr)
    parser.add_argument("--grad-clip", type=float, default=defaults.grad_clip)
    parser.set_defaults(use_traj_loss=True)
    parser.add_argument("--no-traj-loss", action="store_false", dest="use_traj_loss")
    parser.set_defaults(shuffle_windows=True)
    parser.add_argument(
        "--no-shuffle-windows",
        action="store_false",
        dest="shuffle_windows",
        help="Disable per-epoch shuffling of rollout windows.",
    )
    parser.add_argument("--dt-base", type=float, default=defaults.dt_base)
    parser.add_argument(
        "--integrator",
        choices=["rk2", "rk4"],
        default=defaults.integrator,
        help="Time integrator used by trajectory rollout loss/evaluation.",
    )
    parser.add_argument("--batch-size-min", type=int, default=defaults.batch_size_min)
    parser.add_argument("--batch-size-max", type=int, default=defaults.batch_size_max)
    parser.add_argument("--batch-size-schedule", choices=["linear", "exponential"], default=defaults.batch_size_schedule)
    parser.add_argument("--batch-skip", type=int, default=defaults.batch_skip)
    parser.add_argument("--norm-mode", choices=["none", "minmax", "zscore"], default=defaults.norm_mode)
    parser.add_argument("--norm-stats-json", type=str, default=defaults.norm_stats_json)
    parser.add_argument(
        "--loss-weight-mode",
        choices=["none", "variance", "task", "variance_task"],
        default=defaults.loss_weight_mode,
    )
    parser.add_argument("--variance-weights-json", type=str, default=defaults.variance_weights_json)
    parser.add_argument("--task-weights-spec", type=str, default=defaults.task_weights_spec)
    parser.add_argument("--train-progress-every", type=int, default=defaults.train_progress_every)
    parser.add_argument(
        "--best-checkpoint-metric",
        choices=["test_loss", "rmse_norm"],
        default="test_loss",
        help="Metric used to select and restore the best checkpoint.",
    )
    parser.add_argument("--lr-plateau", action="store_true")
    parser.add_argument("--plateau-factor", type=float, default=defaults.plateau_factor)
    parser.add_argument("--plateau-patience", type=int, default=defaults.plateau_patience)
    parser.add_argument("--plateau-threshold", type=float, default=defaults.plateau_threshold)
    parser.add_argument("--plateau-min-lr", type=float, default=defaults.plateau_min_lr)
    parser.add_argument("--plateau-cooldown", type=int, default=defaults.plateau_cooldown)
    parser.add_argument(
        "--max-windows-per-epoch",
        type=int,
        default=defaults.max_windows_per_epoch,
        help="Cap train rollout windows per epoch (0 means no cap).",
    )
    parser.add_argument(
        "--eval-every",
        type=int,
        default=defaults.eval_every,
        help="Evaluate test loss every N epochs (always evaluates last epoch).",
    )
    parser.add_argument(
        "--early-stop-patience",
        type=int,
        default=defaults.early_stop_patience,
        help="Stop after this many evaluated epochs without improvement (0 disables).",
    )
    parser.add_argument(
        "--early-stop-min-delta",
        type=float,
        default=defaults.early_stop_min_delta,
        help="Minimum metric decrease required to reset early-stopping patience.",
    )
    parser.add_argument(
        "--early-stop-min-epochs",
        type=int,
        default=defaults.early_stop_min_epochs,
        help="Do not early-stop before this many epochs have completed.",
    )
    parser.add_argument("--plot-test-rollouts", action="store_true")
    parser.add_argument("--plot-test-rollout-windows", type=int, default=3)
    parser.add_argument("--plot-test-rollout-batch-size", type=int, default=None)


def run_gd_training(
    *,
    model,
    data: StageDataBundle,
    args,
    constraint_kwargs: Dict[str, Any] | None = None,
    eval_param_callback=None,
    best_checkpoint_path: str | None = None,
    restore_best_state: bool = False,
    best_checkpoint_metric_fn=None,
    parameter_projection_callback=None,
    constrained_lrs: Dict[str, float] | None = None,
    named_parameter_lrs: Dict[str, Tuple[str, float]] | None = None,
) -> Dict[str, Any]:
    """Forward common stage data and CLI args into the shared train_gd loop."""
    kwargs = {} if constraint_kwargs is None else dict(constraint_kwargs)
    return train_gd(
        model,
        data.train_traj,
        data.test_traj,
        args.epochs,
        args.lr,
        args.grad_clip,
        use_traj_loss=args.use_traj_loss,
        dt=data.dt,
        integrator=args.integrator,
        batch_size_min=args.batch_size_min,
        batch_size_max=args.batch_size_max,
        batch_size_schedule=args.batch_size_schedule,
        batch_skip=args.batch_skip,
        norm_scale_1_13=data.norm_scale,
        channel_weight_1_13=data.loss_weight,
        train_progress_every=args.train_progress_every,
        shuffle_windows=args.shuffle_windows,
        lr_plateau=args.lr_plateau,
        plateau_factor=args.plateau_factor,
        plateau_patience=args.plateau_patience,
        plateau_threshold=args.plateau_threshold,
        plateau_min_lr=args.plateau_min_lr,
        plateau_cooldown=args.plateau_cooldown,
        max_windows_per_epoch=args.max_windows_per_epoch,
        eval_every=args.eval_every,
        eval_param_callback=eval_param_callback,
        best_checkpoint_path=best_checkpoint_path,
        restore_best_state=restore_best_state,
        best_checkpoint_metric=args.best_checkpoint_metric,
        best_checkpoint_metric_fn=best_checkpoint_metric_fn,
        early_stop_patience=args.early_stop_patience,
        early_stop_min_delta=args.early_stop_min_delta,
        early_stop_min_epochs=args.early_stop_min_epochs,
        parameter_projection_callback=parameter_projection_callback,
        constrained_lrs=constrained_lrs,
        named_parameter_lrs=named_parameter_lrs,
        **kwargs,
    )


def build_gd_meta(args, data: StageDataBundle) -> Dict[str, Any]:
    """Build metadata common to all GD stages."""
    return {
        "train_files": data.train_files,
        "test_files": data.test_files,
        "dt": data.dt,
        "dt_base": args.dt_base,
        "integrator": getattr(args, "integrator", "rk4"),
        "sample_step": args.sample_step,
        "norm_mode": args.norm_mode,
        "norm_stats_json": args.norm_stats_json,
        "loss_weight_mode": args.loss_weight_mode,
        "variance_weights_json": args.variance_weights_json,
        "task_weights_spec": args.task_weights_spec,
        "use_traj_loss": args.use_traj_loss,
        "train_progress_every": args.train_progress_every,
        "best_checkpoint_metric": args.best_checkpoint_metric,
        "shuffle_windows": args.shuffle_windows,
        "lr_plateau": args.lr_plateau,
        "plateau_factor": args.plateau_factor,
        "plateau_patience": args.plateau_patience,
        "plateau_threshold": args.plateau_threshold,
        "plateau_min_lr": args.plateau_min_lr,
        "plateau_cooldown": args.plateau_cooldown,
        "max_windows_per_epoch": args.max_windows_per_epoch,
        "eval_every": args.eval_every,
        "early_stop_patience": args.early_stop_patience,
        "early_stop_min_delta": args.early_stop_min_delta,
        "early_stop_min_epochs": args.early_stop_min_epochs,
        "plot_test_rollouts": bool(args.plot_test_rollouts),
        "plot_test_rollout_windows": args.plot_test_rollout_windows,
        "plot_test_rollout_batch_size": args.plot_test_rollout_batch_size,
    }


# =============================================================================
# 6. Stage A selective body-CLS helpers from new-data workflow
# =============================================================================
OLD_STAGE_B_JSON = os.path.join(
    os.path.dirname(__file__),
    "log",
    "si_awug_crba",
    "stage_b",
    "20260331_134139",
    "stage_b_result.json",
)


class AWUGBodyNoSymmetryModel(AWUGModelFromTex):
    """Variant used only when all ten body coefficients are constrained directly."""

    @staticmethod
    def _apply_body_symmetry(theta_full: torch.Tensor) -> torch.Tensor:
        return theta_full


def body_symmetry_matrix(*, device: str, dtype: torch.dtype) -> torch.Tensor:
    mat = torch.zeros((10, 6), dtype=dtype, device=device)
    # Original y-source mapping:
    # for i in range(6):
    #     mat[i, i] = 1.0
    # mat[6, 2] = 1.0
    # mat[7, 3] = 1.0
    # mat[8, 4] = -1.0
    # mat[9, 5] = -1.0
    mat[0, 0] = 1.0
    mat[1, 1] = 1.0
    mat[2, 2] = 1.0
    mat[3, 3] = 1.0
    mat[4, 4] = -1.0
    mat[5, 5] = -1.0
    mat[6, 2] = 1.0
    mat[7, 3] = 1.0
    mat[8, 4] = 1.0
    mat[9, 5] = 1.0
    return mat


def collect_body_residual_system(
    model: AWUGModelFromTex,
    traj_list: Sequence[torch.Tensor],
    *,
    min_linear_speed: float,
    min_angular_speed: float,
    verbose: bool,
    progress_every: int = 2000,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, int]]:
    X3_rows: List[torch.Tensor] = []
    X4_rows: List[torch.Tensor] = []
    Y3_rows: List[torch.Tensor] = []
    Y4_rows: List[torch.Tensor] = []
    sample_count = 0
    used_count = 0
    skipped_low_excitation = 0

    for traj in traj_list:
        for i in range(traj.shape[0]):
            sample_count += 1
            if verbose and progress_every > 0 and sample_count % progress_every == 0:
                print(f"[stage_a_cls] samples={sample_count} used={used_count}", flush=True)

            eta, v_b, w_b, nu_b, q_s, qd_s, qdd_s, nu_b_dot_meas, fp_code = _parse_state_for_residual(
                model, traj[i]
            )
            linear_speed = float(torch.linalg.norm(v_b).item())
            angular_speed = float(torch.linalg.norm(w_b).item())
            if linear_speed < min_linear_speed and angular_speed < min_angular_speed:
                skipped_low_excitation += 1
                continue
            used_count += 1

            M_q_s, C_nu, M_bs_qdd_s, b_s, g_q = model._accumulate_terms(
                nu_b, eta, q_s, qd_s, qdd_s
            )
            tau_ext_total = M_q_s @ nu_b_dot_meas + C_nu + M_bs_qdd_s + b_s + g_q
            tau_prop = torch.zeros((6, 1), dtype=model.dtype, device=model.device)
            tau_prop[0, 0] = _fp_code_to_F_p(fp_code, model)
            tau_wing_hydro = _compute_tau_wing_hydro_total(model, nu_b, q_s, qd_s)
            tau_body_hydro = tau_ext_total - tau_prop - tau_wing_hydro

            X3_rows.append(model._construct_X3_matrix(v_b, w_b, model.device, model.dtype))
            X4_rows.append(model._construct_X4_matrix(v_b, w_b, model.device, model.dtype))
            Y3_rows.append(tau_body_hydro[:3].reshape(3, 1))
            Y4_rows.append(tau_body_hydro[3:].reshape(3, 1))

    if not X3_rows or not X4_rows:
        raise ValueError("No valid LS rows after excitation filtering")

    stats = {
        "samples": sample_count,
        "used": used_count,
        "skipped_low_excitation": skipped_low_excitation,
    }
    return (
        torch.cat(X3_rows, dim=0),
        torch.cat(X4_rows, dim=0),
        torch.cat(Y3_rows, dim=0),
        torch.cat(Y4_rows, dim=0),
        stats,
    )


def axis_weights(y: torch.Tensor, *, block_dim: int) -> torch.Tensor:
    y_blk = y.reshape(-1, block_dim)
    std_axis = torch.std(y_blk, dim=0, unbiased=False).reshape(1, block_dim)
    return (1.0 / (std_axis + 1e-8)).repeat(y_blk.shape[0], 1).reshape(-1, 1)


def bounded_lsq(
    X: torch.Tensor,
    y: torch.Tensor,
    *,
    lower: np.ndarray,
    upper: np.ndarray,
    block_dim: int,
    weighted: bool,
    col_normalize: bool,
    ridge_lambda: float,
) -> tuple[torch.Tensor, Dict[str, float | int | str | bool]]:
    X_np = X.detach().cpu().numpy()
    y_np = y.detach().cpu().numpy().reshape(-1)

    if weighted:
        w = axis_weights(y, block_dim=block_dim).detach().cpu().numpy().reshape(-1)
        X_np = X_np * w[:, None]
        y_np = y_np * w

    scale = np.ones(X_np.shape[1], dtype=np.float64)
    if col_normalize:
        scale = np.maximum(np.linalg.norm(X_np, axis=0), 1e-12)
        X_solve = X_np / scale.reshape(1, -1)
        lower_solve = lower * scale
        upper_solve = upper * scale
        if ridge_lambda > 0.0:
            ridge = math.sqrt(ridge_lambda) * np.diag(1.0 / scale)
            X_solve = np.vstack([X_solve, ridge])
            y_np = np.concatenate([y_np, np.zeros(X_np.shape[1], dtype=np.float64)])
    else:
        X_solve = X_np
        lower_solve = lower
        upper_solve = upper
        if ridge_lambda > 0.0:
            X_solve = np.vstack([X_solve, math.sqrt(ridge_lambda) * np.eye(X_np.shape[1])])
            y_np = np.concatenate([y_np, np.zeros(X_np.shape[1], dtype=np.float64)])

    result = lsq_linear(
        X_solve,
        y_np,
        bounds=(lower_solve, upper_solve),
        method="trf",
        lsmr_tol="auto",
        max_iter=500,
    )
    theta = result.x / scale
    meta = {
        "success": bool(result.success),
        "status": int(result.status),
        "message": str(result.message),
        "cost": float(result.cost),
        "optimality": float(result.optimality),
        "active_mask_nonzero": int(np.count_nonzero(result.active_mask)),
        "n_iter": int(result.nit),
    }
    return torch.tensor(theta.reshape(-1, 1), dtype=X.dtype, device=X.device), meta


def build_params_with_body(
    theta3: torch.Tensor, theta4: torch.Tensor, stage_json: Dict[str, Any] | None = None
) -> Dict[str, Any]:
    params = clone_base_params()
    if stage_json is not None:
        apply_body_state(params, stage_json)
    params["Theta_3"] = theta3.detach().cpu().numpy()
    params["Theta_4"] = theta4.detach().cpu().numpy()
    return params


def build_body_model(args, *, params: Dict | None = None) -> AWUGModelFromTex:
    if args.body_symmetry_mode == "full_negative_no_symmetry":
        model_params = clone_base_params() if params is None else params
        return AWUGBodyNoSymmetryModel(params=model_params, device=args.device)
    return build_model(device=args.device, params=params)


def build_constrained_forward(
    model: AWUGModelFromTex,
    constraints: Dict[str, ConstrainedRaw],
) -> Callable[[torch.Tensor], Dict[str, torch.Tensor]]:
    """Return a forward function that evaluates with constrained parameter values."""

    def forward_fn(state: torch.Tensor):
        param_map = {name: param for name, param in model.named_parameters()}
        for name, constraint in constraints.items():
            param_map[name] = constraint.value()
        merged = {**param_map, **{name: buf for name, buf in model.named_buffers()}}
        return functional_call(model, merged, (state,))

    return forward_fn


def rollout_loss_value(
    model: AWUGModelFromTex,
    traj_list: List[torch.Tensor],
    *,
    dt: float,
    batch_size: int,
    batch_skip: int,
    norm_scale_1_13: torch.Tensor | None = None,
    channel_weight_1_13: torch.Tensor | None = None,
    forward_fn=None,
    integrator: str = "rk4",
) -> float:
    """Evaluate one trajectory split with the shared windowed rollout loss."""
    return float(
        traj_mse_eval(
            model,
            traj_list,
            dt=dt,
            batch_size=batch_size,
            batch_skip=batch_skip,
            norm_scale_1_13=norm_scale_1_13,
            channel_weight_1_13=channel_weight_1_13,
            forward_fn=forward_fn,
            integrator=integrator,
        )
    )


def rollout_loss_eval(
    model: AWUGModelFromTex,
    data: StageDataBundle,
    args,
    *,
    forward_fn=None,
    integrator: str | None = None,
) -> Dict[str, float]:
    """Evaluate train/test rollout losses with the shared Stage A-E window settings."""
    batch_size = int(args.batch_size_max)
    batch_skip = int(args.batch_skip)
    integrator_name = integrator or getattr(args, "integrator", "rk4")
    return {
        "train_traj_loss": rollout_loss_value(
            model,
            data.train_traj,
            dt=data.dt,
            batch_size=batch_size,
            batch_skip=batch_skip,
            norm_scale_1_13=data.norm_scale,
            channel_weight_1_13=data.loss_weight,
            forward_fn=forward_fn,
            integrator=integrator_name,
        ),
        "test_traj_loss": rollout_loss_value(
            model,
            data.test_traj,
            dt=data.dt,
            batch_size=batch_size,
            batch_skip=batch_skip,
            norm_scale_1_13=data.norm_scale,
            channel_weight_1_13=data.loss_weight,
            forward_fn=forward_fn,
            integrator=integrator_name,
        ),
    }


def signed_summary(theta: torch.Tensor, *, independent_len: int | None = None) -> Dict[str, int | float | bool]:
    flat = theta.detach().cpu().numpy().reshape(-1)
    first = flat if independent_len is None else flat[:independent_len]
    return {
        "min": float(np.min(flat)),
        "max": float(np.max(flat)),
        "num_negative": int(np.sum(flat < 0.0)),
        "num_positive": int(np.sum(flat > 0.0)),
        "first_independent_max": float(np.max(first)),
        "first_independent_all_negative": bool(np.all(first < 0.0)),
    }


def loss_delta(new_loss: Dict[str, float], old_loss: Dict[str, float]) -> Dict[str, Dict[str, float]]:
    out: Dict[str, Dict[str, float]] = {}
    for key, new_val in new_loss.items():
        old_val = old_loss.get(key, float("nan"))
        if math.isfinite(old_val) and abs(old_val) > 1e-18:
            ratio = new_val / old_val
        else:
            ratio = float("nan")
        out[key] = {
            "new_minus_old": float(new_val - old_val),
            "new_over_old": float(ratio),
        }
    return out


def nan_loss_dict() -> Dict[str, float]:
    return {
        "train_traj_loss": float("nan"),
        "test_traj_loss": float("nan"),
    }
