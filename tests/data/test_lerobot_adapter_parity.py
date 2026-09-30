"""The LeRobot adapter returns the same data wherever it runs.

The module writes a small multi-camera LeRobot v3 dataset: two h264 video
cameras, one PNG image camera, episodes of 5, 9 and 14 steps, plus
``next.reward`` and ``next.done``. Each frame is a flat grey level that
depends on the camera and on the row, so a frame from the wrong row, the
wrong episode or the wrong camera shows up as a wrong level.

The tests compare outputs with ``torch.equal`` and check dtypes and shapes,
so "the same" means bit for bit.
"""

from __future__ import annotations

import pickle
import sys

import numpy as np
import pytest
import torch

if sys.version_info < (3, 12):
    pytest.skip('lerobot requires Python 3.12+', allow_module_level=True)

pytest.importorskip('lerobot')

from _lerobot_data import VIDEO_BACKEND, offline_hf, write_dataset  # noqa: E402

from stable_worldmodel.data import LeRobotAdapter  # noqa: E402
from stable_worldmodel.data.formats.lerobot import (  # noqa: E402
    _episode_structure,
)

REPO_ID = 'swm-tests/multi'
FRONT = 'observation.images.front'
WRIST = 'observation.images.wrist'
TOP = 'observation.images.top'
CAMERAS = {FRONT: 'video', WRIST: 'video', TOP: 'image'}
EP_LENGTHS = (5, 9, 14)
EP_OFFSETS = (0, 5, 14)
N_ROWS = sum(EP_LENGTHS)

# Each camera walks through the same 28 levels, shifted, so two cameras
# never show the same level on the same row.
_CAMERA_SHIFT = {FRONT: 0, WRIST: 14, TOP: 7}

KEY_ALIASES = {
    WRIST: 'wrist',
    TOP: 'top',
    'next.reward': 'reward',
    'next.done': 'done',
}


def level(key: str, ep: int, step: int) -> int:
    row = EP_OFFSETS[ep] + step
    return 16 + 8 * ((row + _CAMERA_SHIFT[key]) % N_ROWS)


@pytest.fixture(scope='module', autouse=True)
def _offline_hf(tmp_path_factory):
    """Keep every Hugging Face read offline and inside a temp folder."""
    yield from offline_hf(tmp_path_factory)


@pytest.fixture(scope='module')
def multi_root(tmp_path_factory):
    return write_dataset(
        tmp_path_factory.mktemp('lerobot') / 'ds',
        repo_id=REPO_ID,
        lengths=EP_LENGTHS,
        cameras=CAMERAS,
        level=level,
    )


def _open(root, **kwargs) -> LeRobotAdapter:
    kwargs.setdefault('video_backend', VIDEO_BACKEND)
    kwargs.setdefault('primary_camera_key', FRONT)
    kwargs.setdefault('key_aliases', KEY_ALIASES)
    return LeRobotAdapter(repo_id=REPO_ID, root=root, **kwargs)


def assert_same(expected: dict, actual: dict, where: str = '') -> None:
    """Same keys in the same order, same dtypes, shapes and values."""
    assert list(actual) == list(expected), where
    for key, want in expected.items():
        got = actual[key]
        assert type(got) is type(want), (where, key)
        if isinstance(want, torch.Tensor):
            assert got.dtype == want.dtype, (where, key)
            assert got.shape == want.shape, (where, key)
            assert torch.equal(got, want), (where, key)
        elif isinstance(want, np.ndarray):
            assert got.dtype == want.dtype, (where, key)
            assert np.array_equal(got, want), (where, key)
        else:
            assert got == want, (where, key)


class _Unpicklable:
    def __reduce__(self):
        raise TypeError('this object cannot be pickled')


def test_pickle_round_trip_matches_the_original(multi_root):
    dataset = _open(multi_root, num_steps=2, frameskip=2)
    before = [dataset[i] for i in range(0, len(dataset), 5)]
    # stable-pretraining can attach a trainer that cannot be pickled.
    dataset._trainer = _Unpicklable()

    restored = pickle.loads(pickle.dumps(dataset))

    assert restored._trainer is None
    for i, item in zip(range(0, len(dataset), 5), before):
        assert_same(item, restored[i], f'item {i}')
    for ep in range(len(EP_LENGTHS)):
        assert_same(
            dataset.load_episode(ep), restored.load_episode(ep), f'ep {ep}'
        )
    for col in ('action', 'proprio', 'reward', 'done', 'ep_idx', 'step_idx'):
        np.testing.assert_array_equal(
            restored.get_col_data(col), dataset.get_col_data(col)
        )
        assert restored.get_col_data(col).dtype == (
            dataset.get_col_data(col).dtype
        )


def test_spawn_workers_match_in_process(multi_root):
    from torch.utils.data import DataLoader

    dataset = _open(multi_root, num_steps=2, frameskip=3)
    subset = torch.utils.data.Subset(dataset, list(range(8)))
    in_process = list(DataLoader(subset, batch_size=4))
    spawned = list(
        DataLoader(
            subset,
            batch_size=4,
            num_workers=2,
            multiprocessing_context='spawn',
        )
    )
    assert len(spawned) == len(in_process) == 2
    for b, (want, got) in enumerate(zip(in_process, spawned)):
        assert_same(want, got, f'batch {b}')


# -- Episode structure -------------------------------------------------------


def _legacy_episode_metadata(absolute_episode_index: np.ndarray):
    """Copy of ``LeRobotAdapter._build_episode_metadata`` before this change."""
    abs_ids = absolute_episode_index.astype(np.int64)
    unique_abs, first_idx = np.unique(abs_ids, return_index=True)
    order = np.argsort(first_idx)
    absolute_episode_ids = unique_abs[order]
    counts = np.array(
        [(abs_ids == ep_id).sum() for ep_id in absolute_episode_ids],
        dtype=np.int64,
    )
    local_map = {
        int(abs_id): idx for idx, abs_id in enumerate(absolute_episode_ids)
    }
    local_episode_index = np.array(
        [local_map[int(abs_id)] for abs_id in abs_ids], dtype=np.int64
    )
    step_idx = np.empty_like(local_episode_index)
    for local_ep in range(len(absolute_episode_ids)):
        mask = local_episode_index == local_ep
        step_idx[mask] = np.arange(mask.sum(), dtype=np.int64)
    offsets = np.zeros(len(counts), dtype=np.int64)
    if len(counts) > 1:
        offsets[1:] = np.cumsum(counts[:-1])
    return (
        local_episode_index,
        step_idx,
        counts,
        offsets,
        absolute_episode_ids.astype(np.int64),
    )


def _assert_same_structure(episode_index: np.ndarray) -> None:
    got = _episode_structure(episode_index)
    want = _legacy_episode_metadata(episode_index)
    for name, g, w in zip(got._fields, got, want):
        assert g.dtype == w.dtype, name
        np.testing.assert_array_equal(g, w, err_msg=name)


@pytest.mark.parametrize(
    'episode_index',
    [
        [],
        [7],
        [0, 0, 0],
        [0, 1, 2],
        [3, 3, 1, 1, 1, 8],  # an episode subset keeps LeRobot's row order
        [2, 2, 0, 0, 0, 5, 5],
    ],
)
def test_episode_structure_matches_the_old_code(episode_index):
    _assert_same_structure(np.asarray(episode_index, dtype=np.int64))


def test_episode_structure_matches_the_old_code_on_random_layouts():
    rng = np.random.default_rng(0)
    for _ in range(50):
        n_episodes = int(rng.integers(1, 30))
        ids = rng.permutation(100)[:n_episodes]
        lengths = rng.integers(1, 20, size=n_episodes)
        _assert_same_structure(np.repeat(ids, lengths))


def test_episode_structure_rejects_an_episode_split_in_two():
    with pytest.raises(ValueError, match='episode 0 are not contiguous'):
        _episode_structure(np.array([0, 0, 1, 0]))
