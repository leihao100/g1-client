"""Interactive lower-body (WBC / locomotion) control for the G1.

Drives the four loco control surfaces through g1_client's sibling
``LocoController`` (a wrapper over the Unitree SDK ``LocoClient`` "sport" RPC
service): FSM state transitions, stand height, balance mode, and omnidirectional
move. Modeled on the SDK example ``example/g1/high_level/g1_loco_client_example.py``
but adapted to this repo's entry-script convention (sys.path bootstrap + --iface).

Precondition: the robot's locomotion service is running and the robot is in a
motion mode that accepts loco commands (set via the Unitree app). Do NOT run this
at the same time as an arm_sdk body-lock path — they contend for leg/waist
authority.

WARNING: this makes the robot move/step. Clear the area first, and keep the
remote's emergency stop within reach. Note: `damp` drops all leg torque and the
robot goes limp (it will collapse if not on a rack) — on quit ('q') this tool
instead stops and holds a standing balance posture (FSM 500), never damps.

Run from the repo root:

    python wbc/main.py --iface enp0s31f6
"""

import argparse
import logging
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repo root -> import g1_client

from unitree_sdk2py.core.channel import ChannelFactoryInitialize

from loco_controller import LocoController


log = logging.getLogger("g1_wbc.main")


def _prompt_float(label: str, default: float) -> float:
    raw = input(f"  {label} [{default}]: ").strip()
    if not raw:
        return default
    try:
        return float(raw)
    except ValueError:
        print("  not a number, using default")
        return default


def _do_move(loco: LocoController) -> None:
    vx = _prompt_float("vx (m/s, +fwd)", 0.3)
    vy = _prompt_float("vy (m/s, +left)", 0.0)
    vyaw = _prompt_float("vyaw (rad/s, +left)", 0.0)
    cont = input("  continuous? (y/N): ").strip().lower() == "y"
    loco.move(vx, vy, vyaw, continuous=cont)


def _do_height(loco: LocoController) -> None:
    h = _prompt_float("stand height (m)", 0.7)
    loco.set_stand_height(h)


def _do_balance(loco: LocoController) -> None:
    m = input("  balance mode (0=static, 1=continuous): ").strip()
    loco.set_balance_mode(int(m) if m else 0)


def _do_fsm(loco: LocoController) -> None:
    fsm = input("  raw FSM id: ").strip()
    if fsm:
        loco.set_fsm(int(fsm))


def _do_select_mode(loco: LocoController) -> None:
    name = input("  mode name (normal / ai / advanced) [normal]: ").strip() or "normal"
    code = loco.select_mode(name)
    print(f"  SelectMode({name!r}) -> code {code}")


def _do_status(loco: LocoController) -> None:
    """Read back current mode + FSM / balance / stand height to diagnose no-ops.
    balance/height returning code 7301 means no motion mode is active — run
    'select motion mode' first."""
    print(f"  motion_mode  = {loco.check_mode()}")
    print(f"  fsm_id       = {loco.get_fsm_id()}")
    print(f"  fsm_mode     = {loco.get_fsm_mode()}")
    print(f"  balance_mode = {loco.get_balance_mode()}")
    print(f"  stand_height = {loco.get_stand_height()}  (7301 = unimplemented on this firmware)")
    angles = loco.get_leg_angles()
    if angles is None:
        print("  leg_angles   = (no rt/lowstate yet)")
    else:
        print("  leg_angles(rad):")
        for name, q in angles.items():
            print(f"    {name:14s} {q:+.3f}")


# (label, handler). Handlers take the controller; menu-only entries wrap methods.
MENU = [
    ("select motion mode ... (DO THIS FIRST)", _do_select_mode),
    ("release motion mode (LIMP - will collapse)", lambda l: l.release_mode()),
    ("squat down (id 2)", lambda l: l.squat()),
    ("stand up (id 4)", lambda l: l.stand_up()),
    ("regular locomotion / walk-ready (id 501)", lambda l: l.regular()),
    ("damp (LIMP - will collapse if not on a rack)", lambda l: l.damp()),
    ("zero torque", lambda l: l.zero_torque()),
    ("sit", lambda l: l.sit()),
    ("set raw FSM id ...", _do_fsm),
    ("high stand", lambda l: l.high_stand()),
    ("low stand", lambda l: l.low_stand()),
    ("set stand height ...", _do_height),
    ("set balance mode ...", _do_balance),
    ("move ...", _do_move),
    ("stop move", lambda l: l.stop_move()),
    ("status (read fsm/balance/height)", _do_status),
]


def _print_menu() -> None:
    print("\n=== G1 WBC / loco control ===")
    for i, (label, _) in enumerate(MENU):
        print(f"  {i:2d}  {label}")
    print("  list  reprint this menu     q  quit")


def run(iface: str, timeout: float) -> None:
    ChannelFactoryInitialize(0, iface)
    loco = LocoController(timeout=timeout)
    print("Connected to loco service. Ensure the area is clear.")
    _print_menu()

    while True:
        raw = input("\nEnter id (or 'list' / 'q'): ").strip()
        if raw in ("q", "quit", "exit"):
            break
        if raw == "list" or raw == "":
            _print_menu()
            continue
        try:
            idx = int(raw)
            label, handler = MENU[idx]
        except (ValueError, IndexError):
            print("  no such option")
            continue

        print(f"-> {label}")
        code = handler(loco)
        if isinstance(code, int) and code != 0:
            print(f"  (rpc returned code {code})")
        time.sleep(0.2)

    # Leave the robot standing on exit — stop moving, then return to the standing
    # pose (FSM 4, verified). Deliberately NOT damp(): damping drops all leg torque
    # and the robot collapses, which can damage it when it isn't on a rack.
    print("Stopping and returning to standing pose before exit ...")
    loco.stop_move()
    time.sleep(0.2)
    loco.stand_up()


def main() -> None:
    p = argparse.ArgumentParser(description="Interactive G1 lower-body/loco control")
    p.add_argument("--iface", required=True,
                   help="Network interface to the robot, e.g. enp0s31f6")
    p.add_argument("--timeout", type=float, default=10.0,
                   help="Loco RPC call timeout in seconds (default 10)")
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(name)s %(levelname)s %(message)s")

    print("WARNING: the robot will move/step. Clear the area around it.")
    input("Press Enter to continue...")

    try:
        run(args.iface, args.timeout)
    except KeyboardInterrupt:
        print("\nInterrupted.")


if __name__ == "__main__":
    main()
