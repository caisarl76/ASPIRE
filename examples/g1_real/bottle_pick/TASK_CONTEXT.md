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

## Where grasps work (sim, Dex3, right hand)

Measured in the MuJoCo sim with a side grasp: the hand level, thumb up, palm facing
+y, approaching along +y. The bottle axis is given in the `torso` frame. Each cell
counts full successes (touch, close, lift 3 cm or more, set down upright) out of 3:

| Bottle x \ y | -0.18 | -0.10 | -0.02 | +0.06 |
| --- | --- | --- | --- | --- |
| 0.34 | 2/3 | 3/3 | 0/3 | 1/3 |
| 0.40 | 0/3 (reach edge) | 2/3 | 1/3 | 1/3 |
| 0.46 and farther | out of reach | out of reach | out of reach | out of reach |

- **Best band:** bottle axis x 0.34 to 0.40, y -0.18 to -0.10, right of the torso
  center. Grasp there when you can choose the spot. When the bottle sits outside it,
  ask the operator to move the bottle or the robot rather than forcing a pose.
- **Reach:** from x 0.46 out the palm cannot reach the bottle's side in this
  orientation (`ik_unreachable`). Try the turned-hand orientations from the touch
  recipe before asking for a move.
- **Pressing in:** the first contact is usually the index finger base, and SONIC stops
  the hand 1 to 4 cm short of its target. In the best band, after the first finger
  contact, step another 2 to 3 cm along +y so the palm meets the bottle, then close.
  Near the center (y -0.02 and beyond) that extra push tipped the bottle off the case
  in 4 of 12 runs, so do not press past contact there.
- **Stale positions:** the torso pitches about 9 deg during startup Home and shifts its
  stance between engages, which moves the bottle up to 10 cm in the torso frame.
  Localize after Home and again after any re-engage.
- These numbers are from the sim. On the robot, treat them as a starting guide.

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
   Near contact the hand hides the bottle from `ego_view`; judge the grasp from an
   extra view (`third_view`) when the station has one.
4. Lift a few centimetres. Verify the bottle follows from the returned image and the
   wrist joint change; a closure value or successful motion is not proof.
5. Move to the place spot, lower in bounded steps, open the hand with
   `cap_open_gripper`, back the hand away from the bottle, raise it 10 to 15 cm
   above the box before any other move (the planner does not know the box is there), and
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
