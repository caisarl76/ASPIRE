---
name: g1-bottle-pick-place
description: Fast, scripted G1 right-hand bottle pick, lift and put back (or place) for small models. Run the routine below in one exec call, then judge its report and final images.
---

# G1 bottle pick and lift (scripted)

Use this for "pick up / lift the bottle and put it back" on the G1 (sim or Dex3 robot).
It runs the whole [bottle pick recipe](../../../examples/g1_real/bottle_pick/TASK_CONTEXT.md)
as one script, so you do not plan moves yourself. Your job is only:

1. After the startup Home, run the routine below **once**, in a single exec call. Change only
   `LIFT_M` (the requested lift height in metres) and `QUERY` if the task names another
   bottle. Do not edit the geometry.
2. Read the report it prints (`REPORT {...}`) and look at the two final images.
3. Decide:
   - `ok: true`, the final `third_view` shows the bottle upright on the case and the hand
     clear: call `cap_program_go_home` and report the numbers (lift, placement offset).
   - `ok: false` with `retry: true` and the bottle standing in the images: run the routine
     once more. Never more than one retry.
   - Otherwise (bottle tipped, fell, out of view, or a refusal): keep the arm still and
     report the failure with the report's `reason`. Do not improvise moves.

Expect the bottle to end 1.5 to 4 cm from its start (it shifts in the grip); that is a
success. A `warning` with `ok: true` (a finger lost contact during the lift, the bottle was
still put back) is still a success: do not retry, because a second pick moves the bottle
further from its spot. The routine takes about one minute.

```javascript
const LIFT_M = 0.05, QUERY = "water bottle";
const J = (r) => r.structuredContent ?? JSON.parse((r.content ?? []).find((c) => c.type === "text").text);
const report = { ok: false, retry: false, steps: [] };
const done = (reason, retry = false) => { report.reason = reason; report.retry = retry; text("REPORT " + JSON.stringify(report)); };
const Q = [1, 0, 0, 0];
async function move(p, quat = Q) {
  const j = J(await tools.mcp__cap__cap_move_to_pose({ arm: "right", max_steps: 300,
    target_pose: { position: p, quaternion_wxyz: quat, frame: "torso" },
    strategy: { ik_solver: "mink", trajectory_planner: "interpolation" } }));
  const code = j.job?.result?.error?.code;
  report.steps.push({ move: p.map((v) => +v.toFixed(3)), err_m: j.job?.result?.final_errors?.right ?? null, code: code ?? null });
  if (code && code !== "controller_failed") throw new Error("move refused: " + code + " " + (j.job.result.error.message ?? ""));
  return j;
}
async function locate() {
  const j = J(await tools.mcp__cap__cap_localize_object({ query: QUERY, camera_name: "ego_view", target_frame: "torso" }));
  const g = j.result?.geometry;
  if (!g) throw new Error("bottle not found");
  const [ex, ey, ez] = g.extents, c = g.pose.position;
  if (Math.max(ex, ey, ez) > 0.30 || Math.min(ex, ey, ez) > 0.12) throw new Error("implausible bottle box " + g.extents);
  // The returned center is the visible front surface's: push it 3 cm away from the robot to the
  // axis, and grasp 3 cm below it (just under the body center). Site 6 cm ahead, 5.3 cm right.
  const d = Math.hypot(c[0], c[1]), A = [c[0] + 0.03 * c[0] / d, c[1] + 0.03 * c[1] / d];
  return { c, S: [A[0] + 0.06, A[1] - 0.053, c[2] - 0.03] };
}
try {
  let { c, S } = await locate();
  report.c0 = c.map((v) => +v.toFixed(3));
  await move([S[0], S[1] - 0.08, S[2] + 0.12]);           // above and right of the bottle
  ({ S } = await locate());                                // the torso moved as the arm reached
  await move([S[0], S[1] - 0.08, S[2] + 0.12]);
  await move([S[0], S[1] - 0.08, S[2]]);                   // down beside it
  for (const back of [0.06, 0.04, 0.02, 0]) await move([S[0], S[1] - back, S[2]]);  // in along +y
  const close = J(await tools.mcp__cap__cap_close_gripper({ arm: "right" }));
  const dg = close.job.result.diagnostics;
  report.close = { contact: dg.contact, closure: dg.closure };
  if (!dg.contact) { await tools.mcp__cap__cap_open_gripper({ arm: "right" }); await move([S[0], S[1] - 0.08, S[2] + 0.12]); done("closed on nothing", true); }
  else {
    const H = close.job.result.terminal_state.end_effector_poses.right;  // put back to this pose
    // Did the bottle rise? SAM3 sees it with the hand in view (sim: 4.6-5.8 cm for a good 5 cm
    // lift; 0.4-1.3 cm when an off-center grasp dragged it instead).
    const height = async () => { try { return (await locate()).c[2]; } catch (e) { return null; } };
    const zBefore = await height();
    const up = await move([H.position[0], H.position[1], H.position[2] + LIFT_M], H.quaternion_wxyz);
    const top = up.job.result.terminal_state.end_effector_poses.right.position[2];
    report.hand_rise_m = +(top - H.position[2]).toFixed(3);
    const zAfter = await height();
    report.bottle_rise_m = zBefore === null || zAfter === null ? null : +(zAfter - zBefore).toFixed(3);
    const notLifted = report.bottle_rise_m !== null && report.bottle_rise_m < 0.6 * LIFT_M;
    const hs = J(await tools.mcp__cap__cap_g1_real__get_hand_state({}));
    const fingers = (hs.result ?? hs).right;
    report.finger_change_rad = +Math.max(...fingers.map((q, i) => Math.abs(q - dg.hand_q[i]))).toFixed(3);
    // Sim: a held lift moves the fingers 0.04-0.05 rad from the settled grip; a hand that lost the
    // bottle closes on toward its command (an empty hand ends ~0.8 rad from the holding pose).
    // 0.15-0.45: one finger (usually the index) lost contact but the bottle stayed in the hand
    // and was put back (sim: 0.18 and 0.29, both lifts fine). 0.45 is an estimate for a drop.
    const dropped = report.finger_change_rad >= 0.45;
    if (report.finger_change_rad >= 0.15 && !dropped) report.warning = "a finger lost contact during the lift";
    await move(H.position, H.quaternion_wxyz);               // straight back down
    const open = J(await tools.mcp__cap__cap_open_gripper({ arm: "right" }));
    report.hand_open = +open.job.result.terminal_state.gripper_positions.right.toFixed(3);
    await move([H.position[0], H.position[1] - 0.08, H.position[2]]);         // back away along -y
    const last = await move([H.position[0], H.position[1] - 0.08, H.position[2] + 0.12]);  // then up
    try { const { c: c1 } = await locate(); report.c1 = c1.map((v) => +v.toFixed(3));
      report.offset_m = +Math.hypot(c1[0] - c[0], c1[1] - c[1]).toFixed(3); } catch (e) { report.relocalize = String(e); }
    for (const im of (await tools.mcp__cap__observe({ cameras: ["ego_view", "third_view"] })).content ?? []) if (im.type === "image") image(im);
    if (dropped) done("the bottle slipped out during the lift", true);
    else if (notLifted) done("the bottle did not rise with the hand (grasp off center)", true);
    else { report.ok = true; done("lifted and put back"); }
  }
} catch (e) { done(String(e)); }
```

The routine uses only registered CAP tools. `REPORT` fields: `close.contact`, `hand_rise_m`
(the hand's measured rise), `bottle_rise_m` (SAM3's measured bottle rise; below 0.6 x `LIFT_M`
means the bottle was dragged, not lifted; `null` if a localization failed), `finger_change_rad` (0.15 to 0.45: a finger lost contact but the
bottle stayed; 0.45 and above: it slipped out),
`hand_open`, `offset_m` (the bottle's estimated distance from its start; SAM3 noise is 2 to 3
cm), and every move with its error.
