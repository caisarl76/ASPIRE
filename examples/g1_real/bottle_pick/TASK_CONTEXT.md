---
task: "Pick up the bottle and place it at the target spot."
---

# G1 bottle pick-and-place

## Goal and setup

Move one bottle from its start spot to a requested place spot on a table at roughly
0.75-0.85 m height, using the right arm; keep the left hand open. The bottle must end
upright and standing unsupported at the place spot with the hand released.

Act through `cap_move_to_pose`, `cap_set_gripper`, `cap_g1_real__set_hand_joints`, and
`cap_g1_real__get_hand_state`; current MCP schemas are authoritative. Poses are in the
`torso` frame. Reach is short (about 0.55 m from the shoulder): confirm the bottle and
place spot are inside it before planning, and ask the operator to move the table or
robot rather than forcing a pose.

## Strategy

1. Localize the bottle with `cap_localize_object` in `ego_view`. Reject a mask on the
   wrong object or implausible depth/size. Localize the place spot the same way when
   it is an object.
2. Approach from above in the `torso` frame: a pre-grasp pose above and slightly
   short of the bottle, then a bounded descent to side contact. Keep fingers open.
3. Close with `cap_close_gripper(arm="right")`: it closes in steps, stops when the
   fingers meet the bottle and squeezes lightly. `contact: false` means the hand
   closed on nothing; open with `cap_open_gripper` and correct the approach. For a
   different finger shape use `cap_g1_real__set_hand_joints` (seven radians, Dex3
   order, at most 0.25 rad per message). Finger closure alone is not a grasp.
4. Lift a few centimetres. Verify the bottle follows from the returned image and the
   wrist joint change; a closure value or successful motion is not proof.
5. Move to the place spot, lower in bounded steps, open the hand with
   `cap_open_gripper`, retract, and
   inspect that the bottle stands.
6. Report the result and the evidence.

## Recovery

A refusal "outside the configured limits" establishes no movement; change the
target or staging pose. If the bottle slips or tips, keep the arm still, relocate the
bottle from a fresh view, and regrasp with a different closure or approach. Do not
loosen tolerances or push through suspected contact.

## Completion and safety

Finish when the bottle stands at the place spot and the hand is released and clear,
shown by a fresh view. **Do not call Home while holding the bottle**: place and
release first, since Home opens the grippers.
