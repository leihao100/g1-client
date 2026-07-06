"""G1 dual-arm FK/IK for the EEF-space openpi client.

Self-contained (pinocchio + numpy only — the PyPI `pin` wheel is enough; no
casadi/meshcat/teleop imports). The kinematic definition matches BOTH
xr_teleoperate's G1_29_ArmIK and the dataset converter
~/unitree/data/joint_to_eef.py exactly:

  * reduced 14-dof model: everything but the 14 arm joints locked at 0
    (pelvis frame, waist locked — same assumption the data was recorded under)
  * EEF frames 'L_ee'/'R_ee' = left/right_wrist_yaw_joint + 0.05 m local x
  * joint order == ARM_JOINTS in g1_client/arm_controller.py
    (L shoulder p/r/y, elbow, wrist r/p/y, then R same) — identical to the
    reduced model's q order, no remapping needed
  * EEF pose layout: (x, y, z, qx, qy, qz, qw), pelvis frame

IK is damped-least-squares over both arms, warm-started from the previous
solution. Unlike teleop's IPOPT optimizer it has no ‖q‖ regularizer — the warm
start keeps the redundant elbow DOF on the training-data branch, and targets
predicted by a policy trained on FK data are (near-)exactly reachable, so a
few Gauss-Newton steps per tick reach sub-mm residuals.
"""

import numpy as np
import pinocchio as pin

DEFAULT_URDF = "/home/ur3-exp/unitree/xr_teleoperate/assets/g1/g1_body29_hand14.urdf"
DEFAULT_ASSETS = "/home/ur3-exp/unitree/xr_teleoperate/assets/g1/"
EE_OFFSET = 0.05  # meters along wrist-yaw local x, same as G1_29_ArmIK

ARM_JOINT_NAMES = [
    "left_shoulder_pitch_joint", "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint", "left_elbow_joint",
    "left_wrist_roll_joint", "left_wrist_pitch_joint", "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint", "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint", "right_elbow_joint",
    "right_wrist_roll_joint", "right_wrist_pitch_joint", "right_wrist_yaw_joint",
]


def _xyzquat_to_se3(pose7: np.ndarray) -> pin.SE3:
    """(x,y,z,qx,qy,qz,qw) -> SE3, normalizing the quaternion (model output
    is not guaranteed unit-norm)."""
    pose7 = np.asarray(pose7, dtype=np.float64)
    q = pose7[3:7]
    n = np.linalg.norm(q)
    if n < 1e-6:
        raise ValueError(f"degenerate quaternion in EEF pose: {pose7}")
    return pin.XYZQUATToSE3(np.concatenate([pose7[:3], q / n]))


class G1DualArmKinematics:
    def __init__(self, urdf_path: str = DEFAULT_URDF, assets_dir: str = DEFAULT_ASSETS):
        robot = pin.RobotWrapper.BuildFromURDF(urdf_path, assets_dir)
        lock = [n for n in robot.model.names[1:] if n not in ARM_JOINT_NAMES]
        self.model = robot.buildReducedRobot(
            list_of_joints_to_lock=lock,
            reference_configuration=np.zeros(robot.model.nq),
        ).model
        if self.model.nq != 14:
            raise RuntimeError(f"expected 14-dof reduced model, got nq={self.model.nq}")
        # q order must equal ARM_JOINTS order; the G1 URDF satisfies this
        # (left arm subtree precedes right arm subtree). Guard it anyway.
        if list(self.model.names[1:]) != ARM_JOINT_NAMES:
            raise RuntimeError(f"reduced-model joint order mismatch: {list(self.model.names[1:])}")
        for side, joint in (("L_ee", "left_wrist_yaw_joint"), ("R_ee", "right_wrist_yaw_joint")):
            self.model.addFrame(pin.Frame(
                side, self.model.getJointId(joint), 0,
                pin.SE3(np.eye(3), np.array([EE_OFFSET, 0.0, 0.0])),
                pin.FrameType.OP_FRAME))
        self.data = self.model.createData()
        self.l_id = self.model.getFrameId("L_ee")
        self.r_id = self.model.getFrameId("R_ee")
        self.q_lo = self.model.lowerPositionLimit
        self.q_hi = self.model.upperPositionLimit

    def fk(self, q14: np.ndarray):
        """14 joint angles -> (left, right) EEF poses, each (x,y,z,qx,qy,qz,qw)."""
        pin.framesForwardKinematics(self.model, self.data, np.asarray(q14, dtype=np.float64))
        return (pin.SE3ToXYZQUAT(self.data.oMf[self.l_id]),
                pin.SE3ToXYZQUAT(self.data.oMf[self.r_id]))

    def solve_ik(self, left_pose7, right_pose7, q_init,
                 max_iters: int = 20, tol: float = 1e-5, damping: float = 1e-8):
        """Damped-least-squares IK for both arms.

        Returns (q14, pos_err_m) where pos_err_m is the worst residual EEF
        position error of the two arms. Warm-start q_init with the previous
        solution (or the measured arm q) — that is what keeps the redundant
        DOF continuous.
        """
        target_l = _xyzquat_to_se3(left_pose7)
        target_r = _xyzquat_to_se3(right_pose7)
        q = np.clip(np.asarray(q_init, dtype=np.float64).copy(), self.q_lo, self.q_hi)
        for _ in range(max_iters):
            pin.framesForwardKinematics(self.model, self.data, q)
            err_l = pin.log(self.data.oMf[self.l_id].actInv(target_l)).vector
            err_r = pin.log(self.data.oMf[self.r_id].actInv(target_r)).vector
            err = np.concatenate([err_l, err_r])
            if np.linalg.norm(err) < tol:
                break
            pin.computeJointJacobians(self.model, self.data, q)
            J = np.vstack([
                pin.getFrameJacobian(self.model, self.data, self.l_id, pin.ReferenceFrame.LOCAL),
                pin.getFrameJacobian(self.model, self.data, self.r_id, pin.ReferenceFrame.LOCAL),
            ])  # (12, 14)
            dq = J.T @ np.linalg.solve(J @ J.T + damping * np.eye(12), err)
            q = np.clip(q + dq, self.q_lo, self.q_hi)
        pin.framesForwardKinematics(self.model, self.data, q)
        pos_err = max(
            np.linalg.norm(self.data.oMf[self.l_id].translation - target_l.translation),
            np.linalg.norm(self.data.oMf[self.r_id].translation - target_r.translation),
        )
        return q, pos_err
