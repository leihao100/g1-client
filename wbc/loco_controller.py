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

import logging

from unitree_sdk2py.g1.loco.g1_loco_client import LocoClient


log = logging.getLogger("g1_wbc.loco")


# ---- FSM ids (from g1_loco_client.py's named helpers) ----
class Fsm:
    ZERO_TORQUE = 0     # LocoClient.ZeroTorque()
    DAMP = 1            # LocoClient.Damp()        — soft/limp, safe to leave in
    SIT = 3            # LocoClient.Sit()
    START = 500         # LocoClient.Start()       — main locomotion control
    LIE_TO_STANDUP = 702    # LocoClient.Lie2StandUp()  (robot face-up, hard flat floor)
    SQUAT_STANDUP = 706     # LocoClient.Squat2StandUp() / StandUp2Squat() (same id)


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


class LocoController:
    """High-level lower-body control for the G1 via the loco RPC service."""

    def __init__(self, timeout: float = 10.0):
        self.client = LocoClient()
        self.client.SetTimeout(timeout)
        self.client.Init()

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

    def start(self) -> int:
        """Enter main locomotion control (must be here before move())."""
        return self.set_fsm(Fsm.START)

    def squat_to_standup(self) -> int:
        return self.set_fsm(Fsm.SQUAT_STANDUP)

    def standup_to_squat(self) -> int:
        return self.set_fsm(Fsm.SQUAT_STANDUP)

    def lie_to_standup(self) -> int:
        """Stand up from lying down. Requires robot face-up on hard, flat, rough
        ground — same precondition as the SDK example."""
        return self.set_fsm(Fsm.LIE_TO_STANDUP)

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
