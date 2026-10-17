state = get_robot_state()
start = state.joint_positions["right"].copy()
target = start.copy()
# Right elbow (index 3 of the seven arm joints): +0.1 rad, then back.
target[3] = target[3] + 0.1
out = move_to_joints(target, arm="right")
back = move_to_joints(start, arm="right")
final = get_robot_state()
error = abs(float(final.joint_positions["right"][3] - start[3]))
result = {"out": out, "back": back, "return_error_rad": error}
