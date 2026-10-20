---
task: "Touch the water bottle with the right palm."
---

# G1 palm touch (rubber hand)

## Goal and setup

Bring the right palm into light contact with the side of the standing water bottle,
then back away. The bottle must stay upright where it was. This station (PC2
`192.168.0.223`) has rubber hands with no motors: `cap_set_gripper`,
`cap_g1_real__set_hand_joints` and `cap_g1_real__get_hand_state` move and report
nothing real, so do not use them and do not treat them as evidence. Leave the left
arm where it is.

Act through `cap_localize_object`, `cap_move_to_pose` and `observe`; current MCP
schemas are authoritative. Poses are in the `torso` frame.

## Hand geometry

`cap_move_to_pose` places the `right_grasp_site`, 0.19 m along the wrist axis, which
is past the end of this hand, not on the palm.

The hand is held level with the thumb up, turned by a yaw `t` about torso +z
(`t <= 0` turns the fingers toward the right). With `t` in radians:

- finger direction `f = (cos t, sin t, 0)`, palm normal `n = (-sin t, cos t, 0)`;
- orientation `quaternion_wxyz = [cos(t/2), 0, 0, sin(t/2)]`:
  `t = 0`: `[1, 0, 0, 0]` (handshake: fingers forward, palm facing +y);
  `t = -30 deg`: `[0.966, 0, 0, -0.259]`; `t = -45 deg`: `[0.924, 0, 0, -0.383]`;
- the palm contact point is the site position `- 0.08 f + 0.03 n`, so to put the
  palm on a point `C`, command the site at `C + 0.08 f - 0.03 n`.

Turning the hand right (`t` negative) brings the site closer to the palm in x and
reaches farther; it touches the bottle's rear-right side instead of its right side.

## Strategy

1. Localize the bottle in `ego_view` (`"water bottle"`). Reject a mask on the wrong
   object or an implausible depth or size. Take the radius `r` from its width (about
   0.03 to 0.04 m). Depth sees only the near surface, so the returned center is about
   one radius too close: the bottle axis is at `A = (cx + r, cy, z)`.
2. Pick a contact height `z` on the bottle's upper half, below the cap, and at least
   0.06 m above the surface it stands on (the hand reaches 0.043 m below the site).
   A higher contact point reaches farther.
3. For `t = 0`: contact point `C = A - r n`, contact site `S = C + 0.08 f - 0.03 n`,
   pre-contact site `P = S - 0.08 n` (8 cm back along the palm normal).
4. Move to `P` in that orientation. If the hand starts low or close to the body, first
   raise it above `z`; the planning model has no table and no legs. **Reach is decided
   by trying, not by a distance rule**: an `ik_unreachable` or "outside the configured
   limits" refusal moves nothing. On a refusal retry with `t = -30 deg`, then
   `t = -45 deg`, then a contact point 2 to 3 cm higher. As a guide, IK reaches a
   bottle axis about 0.46 m in front of the torso at 0.10 m contact height. Ask the
   operator to move the bottle only after all of these are refused.
5. Approach along `+n` with steps of at most 2 cm toward `S`, and look (`observe`)
   after each step. Stop at the first step where the image shows the palm on the
   bottle or the bottle moving. Never push past contact.
6. Retract 8 cm along `-n`, observe, and report.

## Tracking error

SONIC tracks arm targets with a 2 to 4 cm end-effector error, which the operator has
accepted for now. `cap_move_to_pose` already corrects in up to three rounds, so one
move can take several seconds. The returned position error describes the site, not
contact. The image decides whether the palm touches. If the hand stops short with
the bottle undisturbed after the final step, take at most two more 1 cm steps.

## Recovery and completion

If the bottle tips or slides, stop, keep the arm still, and report; do not chase it.
A refusal "outside the configured limits" moved nothing; pick another contact height.
Finish when a fresh view shows the palm touched the bottle, the hand is retracted,
and the bottle stands where it was. Call Home only when asked.
