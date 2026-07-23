"""Lower-body / locomotion controller for the G1.

Thin wrapper over the Unitree SDK ``LocoClient`` (the "sport" RPC service,
``unitree_sdk2py/g1/loco/g1_loco_client.py``) that groups the exposed high-level
commands into the four control surfaces the WBC entry script drives:

  - FSM     : discrete state-machine transitions (damp, start, sit, stand-up, ...)
  - height  : stand height (high / low / an explicit value)
  - balance : balance mode (static vs continuous-gait)
  - move    : omnidirectional velocity (vx, vy, vyaw)

This does NOT publish ``LowCmd_`` on ``rt/arm_sdk`` like ``ArmController`` — it
calls the loco RPC service. So it must NOT run at the same time as the arm_sdk
body-lock path (they fight over leg/waist authority). The robot must already be
in a locomotion-capable motion mode, set via the Unitree app.

The velocity/height clamps below are conservative software limits on top of
whatever the firmware enforces — tighten, don't widen, without a specific reason.
"""

import json
import logging

from unitree_sdk2py.core.channel import ChannelSubscriber
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowState_
from unitree_sdk2py.g1.loco.g1_loco_client import LocoClient
from unitree_sdk2py.comm.motion_switcher.motion_switcher_client import (
    MotionSwitcherClient,
)
from unitree_sdk2py.g1.loco.g1_loco_api import (
    ROBOT_API_ID_LOCO_GET_FSM_ID,
    ROBOT_API_ID_LOCO_GET_FSM_MODE,
    ROBOT_API_ID_LOCO_GET_BALANCE_MODE,
    ROBOT_API_ID_LOCO_GET_STAND_HEIGHT,
)


log = logging.getLogger("g1_wbc.loco")


# ---- FSM ids VERIFIED on this robot's firmware (2026-07) by direct testing.
# These differ from the ids in the bundled g1_loco_client.py, which target an
# OLDER firmware: its Squat2StandUp/StandUp2Squat=706 and Lie2StandUp=702 are
# NOT recognized here (sending them is a no-op). Use the numeric ids below. ----
class Fsm:
    ZERO_TORQUE = 0   # verified
    DAMP = 1          # verified — soft/limp; robot COLLAPSES if not on a rack
    SQUAT = 2         # verified — squat down
    SIT = 3           # from SDK, not re-verified on this firmware
    GET_READY = 4     # verified — slowly returns toward the standing pose (stand up)
    START = 500       # regular locomotion, 1-DoF waist (docs note: jittery)
    REGULAR = 501     # regular locomotion, 3-DoF waist (smooth); the remote's walk state


# ---- Balance modes (firmware convention; confirm against your firmware doc) ----
class Balance:
    STATIC = 0       # stand in place, no stepping
    CONTINUOUS = 1   # continuous-gait / stepping balance, ready to accept move


# ---- Stand-height range (metres). HighStand/LowStand let the firmware clamp
# to its own extremes via UINT32 sentinels; explicit values are clipped here. ----
STAND_HEIGHT_MIN = 0.5
STAND_HEIGHT_MAX = 0.8

# ---- Conservative velocity clamps (firmware also clamps). ----
VX_LIMIT = 0.6     # m/s, +forward
VY_LIMIT = 0.4     # m/s, +left
VYAW_LIMIT = 0.6   # rad/s, +left turn


def _clip(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


# Leg joint indices on rt/lowstate (same order as arm_controller.G1JointIndex).
# These pitch joints in the sagittal plane are what actually set the crouch depth.
LEG_PITCH_JOINTS = {
    "L_hip_pitch": 0, "L_knee": 3, "L_ankle_pitch": 4,
    "R_hip_pitch": 6, "R_knee": 9, "R_ankle_pitch": 10,
}
kTopicLowState = "rt/lowstate"


class LocoController:
    """High-level lower-body control for the G1 via the loco RPC service."""

    def __init__(self, timeout: float = 10.0):
        self.client = LocoClient()
        self.client.SetTimeout(timeout)
        self.client.Init()

        # The loco FSM only accepts SetFsmId / SetStandHeight / SetBalanceMode
        # once a motion mode is selected (otherwise every loco call returns
        # 7301 LOCOSTATE_NOT_AVAILABLE). Mode selection goes through this
        # service, NOT SetFsmId. Equivalent to picking a mode in the Unitree app.
        self.msc = MotionSwitcherClient()
        self.msc.SetTimeout(timeout)
        self.msc.Init()

        # Lazily-opened lowstate subscriber. GET_STAND_HEIGHT is unimplemented on
        # this firmware (returns 7301), so the only way to see how low the robot
        # is standing is the leg joint angles read straight off rt/lowstate.
        self._lowstate_sub = None

    # ---------------- leg pose read-back (crouch depth) ----------------

    def get_leg_angles(self):
        """Return {joint: radians} for the six sagittal leg pitch joints, or None
        if no lowstate has arrived yet. Knee angle is the clearest crouch measure:
        larger |knee| == squatting lower. This is the reliable substitute for the
        dead GET_STAND_HEIGHT RPC."""
        if self._lowstate_sub is None:
            self._lowstate_sub = ChannelSubscriber(kTopicLowState, LowState_)
            self._lowstate_sub.Init()
        msg = self._lowstate_sub.Read()
        if msg is None:
            return None
        return {name: msg.motor_state[idx].q for name, idx in LEG_PITCH_JOINTS.items()}

    # ---------------- motion mode (must be selected before anything else) ----

    def check_mode(self):
        """Return (code, info_dict) — current active motion mode, or None."""
        return self.msc.CheckMode()

    def select_mode(self, name: str = "normal"):
        """Activate a motion mode so the loco FSM comes alive. Common names:
        'normal' (classic locomotion), 'ai' (RL controller, used by the arm_sdk
        path), 'advanced'. Returns the RPC code (0 == ok)."""
        log.info("SelectMode(%r)", name)
        code, _ = self.msc.SelectMode(name)
        return code

    def release_mode(self):
        """Release the active motion mode (robot drops out of loco control)."""
        code, _ = self.msc.ReleaseMode()
        return code

    # ---------------- read-back (diagnostics) ----------------
    # The G1 python LocoClient registers these GET apis but exposes no getters,
    # so we call the underlying RPC directly (same pattern as the H2 client).

    def _get(self, api_id: int):
        """Return (code, value). value is None if the RPC failed (code != 0)."""
        code, data = self.client._Call(api_id, json.dumps({}))
        if code != 0:
            return code, None
        return code, json.loads(data).get("data")

    def get_fsm_id(self):
        return self._get(ROBOT_API_ID_LOCO_GET_FSM_ID)

    def get_fsm_mode(self):
        return self._get(ROBOT_API_ID_LOCO_GET_FSM_MODE)

    def get_balance_mode(self):
        return self._get(ROBOT_API_ID_LOCO_GET_BALANCE_MODE)

    def get_stand_height(self):
        return self._get(ROBOT_API_ID_LOCO_GET_STAND_HEIGHT)

    # ---------------- FSM ----------------

    def set_fsm(self, fsm_id: int) -> int:
        """Send a raw FSM id. Returns the RPC return code (0 == ok)."""
        log.info("SetFsmId(%d)", fsm_id)
        return self.client.SetFsmId(fsm_id)

    def zero_torque(self) -> int:
        return self.set_fsm(Fsm.ZERO_TORQUE)

    def damp(self) -> int:
        return self.set_fsm(Fsm.DAMP)

    def sit(self) -> int:
        return self.set_fsm(Fsm.SIT)

    def squat(self) -> int:
        """Squat down (verified id 2)."""
        return self.set_fsm(Fsm.SQUAT)

    def stand_up(self) -> int:
        """Return toward the standing pose (verified id 4). Slow, safe recovery
        from a squat; never goes limp."""
        return self.set_fsm(Fsm.GET_READY)

    def regular(self) -> int:
        """Regular locomotion, 3-DoF waist (id 501) — the walk-ready state where
        move() works. Preferred over start()/500, which is jittery."""
        return self.set_fsm(Fsm.REGULAR)

    def start(self) -> int:
        """Regular locomotion, 1-DoF waist (id 500). Prefer regular()/501."""
        return self.set_fsm(Fsm.START)

    # ---------------- height ----------------

    def high_stand(self) -> int:
        log.info("HighStand")
        return self.client.HighStand()

    def low_stand(self) -> int:
        log.info("LowStand")
        return self.client.LowStand()

    def set_stand_height(self, height_m: float) -> int:
        """Set an explicit stand height in metres, clipped to
        [STAND_HEIGHT_MIN, STAND_HEIGHT_MAX]."""
        h = _clip(height_m, STAND_HEIGHT_MIN, STAND_HEIGHT_MAX)
        if h != height_m:
            log.warning("stand height %.3f clipped to %.3f", height_m, h)
        log.info("SetStandHeight(%.3f)", h)
        return self.client.SetStandHeight(h)

    # ---------------- balance ----------------

    def set_balance_mode(self, mode: int) -> int:
        log.info("SetBalanceMode(%d)", mode)
        return self.client.SetBalanceMode(mode)

    def balance_static(self) -> int:
        return self.set_balance_mode(Balance.STATIC)

    def balance_continuous(self) -> int:
        return self.set_balance_mode(Balance.CONTINUOUS)

    # ---------------- move ----------------

    def move(self, vx: float, vy: float, vyaw: float,
             continuous: bool = False) -> int:
        """Omnidirectional velocity command. Velocities are clipped to the
        conservative limits above. With continuous=True the command holds until
        superseded/stopped (SDK uses a ~10-day duration); otherwise it lasts ~1s.
        """
        vx = _clip(vx, -VX_LIMIT, VX_LIMIT)
        vy = _clip(vy, -VY_LIMIT, VY_LIMIT)
        vyaw = _clip(vyaw, -VYAW_LIMIT, VYAW_LIMIT)
        log.info("Move(vx=%.3f, vy=%.3f, vyaw=%.3f, continuous=%s)",
                 vx, vy, vyaw, continuous)
        return self.client.Move(vx, vy, vyaw, continuous)

    def stop_move(self) -> int:
        log.info("StopMove")
        return self.client.StopMove()
