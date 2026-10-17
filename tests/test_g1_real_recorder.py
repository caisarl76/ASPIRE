import time

import numpy as np

from cap_harness.g1_real.body_client import FakeBodyClient
from cap_harness.g1_real.config import load_g1_station_config
from cap_harness.g1_real.env import G1Env
from cap_harness.yam_real.cameras import SyntheticCamera
from cap_harness.yam_real.recorder import YamEpisodeRecorder


def test_recorder_writes_seven_joint_arrays_for_g1(tmp_path):
    cfg = load_g1_station_config("g1-lab")
    env = G1Env(cfg, FakeBodyClient(), camera=SyntheticCamera())
    env.engage()
    rec = YamEpisodeRecorder(
        tmp_path / "episode", video_dir=tmp_path / "videos", fps=50, cameras=(), task="t"
    )
    rec.start(env, meta={})
    env.go_home(duration_s=0.3)
    time.sleep(0.2)
    rec.finalize()
    env.close()
    assert np.load(tmp_path / "episode" / "left-joint_pos.npy").shape[1] == 7
    assert np.load(tmp_path / "episode" / "action-left-pos.npy").shape[1] == 8
