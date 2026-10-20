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

- **Handshake orientation**, `quaternion_wxyz: [1, 0, 0, 0]`: fingers point forward
  (+x), thumb up (+z), palm facing left (+y), toward the body midline.
- In that orientation the palm contact point is the site position plus
  `(-0.08, +0.03, 0.0)` m. To put the palm on a point `C`, command the site at
  `C + (0.08, -0.03, 0.0)`.

## Strategy

1. Localize the bottle in `ego_view` (`"water bottle"`). Take its center `(cx, cy, cz)`
   and radius `r` from its width (about 0.03 to 0.04 m). Reject a mask on the wrong
   object or an implausible depth or size.
2. Check reach before moving. It is reliable for a bottle center at x 0.25 to 0.30 m
   and y 0.0 to -0.10 m, and to x 0.40 m when the contact point is at or above
   the torso origin height (z >= 0). Otherwise ask the operator to move the bottle.
3. Contact point `C = (cx, cy - r, cz)`: the bottle's right side, because the right
   palm faces +y. Contact at mid-height or a little above, never near the table.
4. Pre-contact: site at `C + (0.08, -0.11, 0.0)`, 8 cm right of contact, in the
   handshake orientation. If the hand starts low or close to the body, first raise
   it above the contact height; the planning model has no table and no legs.
5. Approach in +y with steps of at most 2 cm toward `C + (0.08, -0.03, 0.0)`, and look
   (`observe`) after each step. Stop at the first step where the image shows the palm
   on the bottle or the bottle moving. Never push past contact.
6. Retract 8 cm in -y, observe, and report.

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
