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

from stable_worldmodel.data import GoalDataset, LeRobotAdapter  # noqa: E402
from stable_worldmodel.data.formats import lerobot as lerobot_format  # noqa: E402
from stable_worldmodel.data.formats.lerobot import (  # noqa: E402
    _arrow_to_numpy,
    _column_to_numpy,
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


# -- Column reads -------------------------------------------------------------

# Every table column of the test dataset, under an alias.
TABLE_ALIASES = {
    **KEY_ALIASES,
    'timestamp': 'timestamp',
    'frame_index': 'frame_index',
    'episode_index': 'episode_index',
    'index': 'index',
    'task_index': 'task_index',
}
TABLE_COLUMNS = {
    'action': 'action',
    'proprio': 'observation.state',
    **{
        alias: native
        for native, alias in TABLE_ALIASES.items()
        if native not in CAMERAS
    },
}


def _legacy_column(adapter: LeRobotAdapter, native_key: str) -> np.ndarray:
    """The column as the old code read it: row by row through ``datasets``."""
    return _column_to_numpy(adapter.dataset.hf_dataset[native_key])


@pytest.mark.parametrize('episodes', [None, [2, 0]])
def test_columns_match_the_old_row_by_row_read(multi_root, episodes):
    adapter = _open(multi_root, episodes=episodes, key_aliases=TABLE_ALIASES)
    for alias, native in TABLE_COLUMNS.items():
        got = adapter.get_col_data(alias)
        want = _legacy_column(adapter, native)
        assert got.dtype == want.dtype, alias
        assert got.shape == want.shape, alias
        np.testing.assert_array_equal(got, want, err_msg=alias)
        assert got.flags.writeable, alias


def test_columns_without_a_numpy_layout_keep_the_old_read(
    multi_root, monkeypatch
):
    monkeypatch.setattr(lerobot_format, '_numpy_layout', lambda _type: None)
    adapter = _open(multi_root, key_aliases=TABLE_ALIASES)
    for alias, native in TABLE_COLUMNS.items():
        got = adapter.get_col_data(alias)
        want = _legacy_column(adapter, native)
        assert got.dtype == want.dtype, alias
        np.testing.assert_array_equal(got, want, err_msg=alias)


def test_arrow_to_numpy_matches_the_datasets_transform():
    """Same values and dtypes as ``hf_transform_to_torch``, type by type."""
    import datasets
    from lerobot.datasets.io_utils import hf_transform_to_torch

    n = 6
    rng = np.random.default_rng(0)
    values = {
        'f16': rng.standard_normal(n).astype(np.float16),
        'f32': rng.standard_normal(n).astype(np.float32),
        'f64': rng.standard_normal(n),
        'i8': rng.integers(-100, 100, n).astype(np.int8),
        'i32': rng.integers(-1000, 1000, n).astype(np.int32),
        'u8': rng.integers(0, 255, n).astype(np.uint8),
        'u32': rng.integers(0, 2**31, n).astype(np.uint32),
        'flag': rng.integers(0, 2, n).astype(bool),
        'vec': rng.standard_normal((n, 3)).astype(np.float32),
        'grid': rng.integers(0, 9, (n, 2, 2)).astype(np.int32),
    }
    value = datasets.Value
    features = datasets.Features(
        {
            'f16': value('float16'),
            'f32': value('float32'),
            'f64': value('float64'),
            'i8': value('int8'),
            'i32': value('int32'),
            'u8': value('uint8'),
            'u32': value('uint32'),
            'flag': value('bool'),
            'vec': datasets.Sequence(value('float32'), length=3),
            'grid': datasets.Sequence(
                datasets.Sequence(value('int32'), length=2), length=2
            ),
        }
    )
    part = datasets.Dataset.from_dict(
        {k: v.tolist() for k, v in values.items()}, features=features
    )
    # Two parts, so the Arrow columns have more than one chunk.
    table = datasets.concatenate_datasets([part, part])
    table.set_transform(hf_transform_to_torch)
    for key in values:
        column = table.data.column(key)
        assert column.num_chunks == 2, key
        want = _column_to_numpy(table[key])
        for got in (
            _arrow_to_numpy(column),
            _arrow_to_numpy(column.slice(0, 2 * n)),
        ):
            assert got.dtype == want.dtype, key
            assert got.shape == want.shape, key
            np.testing.assert_array_equal(got, want, err_msg=key)
        # A slice across the chunk boundary.
        np.testing.assert_array_equal(
            _arrow_to_numpy(column.slice(n - 2, 4)), want[n - 2 : n + 2]
        )


def test_arrow_to_numpy_leaves_other_types_to_datasets():
    import pyarrow as pa

    assert _arrow_to_numpy(pa.array(['a', 'b'])) is None
    assert _arrow_to_numpy(pa.array([[1.0], [2.0, 3.0]])) is None
    assert _arrow_to_numpy(pa.array([1, 2], type=pa.uint64())) is None
    assert _arrow_to_numpy(pa.array([1.0, None])) is None
    assert (
        _arrow_to_numpy(
            pa.array([[1.0, 2.0], None], type=pa.list_(pa.float32(), 2))
        )
        is None
    )


# -- Windows: the adapter against a copy of the old code ---------------------


class _LegacyAdapter(LeRobotAdapter):
    """The adapter with its window code from before this change.

    For each window shape, the old code built one more ``LeRobotDataset``
    with ``delta_timestamps`` and read the window from its row. The two
    methods below are copies of the old ``_window_dataset`` and
    ``_load_slice``. Window datasets are shared between instances with the
    same settings, only to keep the test fast.
    """

    _windows: dict = {}

    def _window_dataset(self, observation_indices, action_indices):
        from lerobot.datasets.lerobot_dataset import LeRobotDataset

        delta_timestamps = {}
        for key in self._keys:
            if key in self._SYNTHETIC_COLUMNS:
                continue
            native_key = self._alias_to_native.get(key)
            if native_key is None:
                continue
            indices = (
                action_indices if key == 'action' else observation_indices
            )
            delta_timestamps[native_key] = [
                float(idx) / self._fps for idx in indices
            ]
        cache_key = (
            str(self.root),
            None if self.episodes is None else tuple(self.episodes),
            repr(sorted(self._lerobot_kwargs.items())),
            repr(sorted(delta_timestamps.items())),
        )
        if cache_key not in self._windows:
            self._windows[cache_key] = LeRobotDataset(
                repo_id=self.repo_id,
                root=self.root,
                episodes=self.episodes,
                image_transforms=None,
                delta_timestamps=delta_timestamps or None,
                **self._lerobot_kwargs,
            )
        return self._windows[cache_key]

    def _load_slice(self, ep_idx, start, end):
        g_start = int(self.offsets[ep_idx] + start)
        length = int(end - start)
        obs_indices = tuple(range(0, length, self.frameskip))
        action_indices = tuple(range(length))
        row = dict(self._window_dataset(obs_indices, action_indices)[g_start])
        steps = {}
        for key in self._keys:
            if key == 'ep_idx':
                data = torch.full(
                    (len(obs_indices),),
                    int(self._cache['ep_idx'][g_start]),
                    dtype=torch.int64,
                )
            elif key == 'step_idx':
                data = torch.as_tensor(
                    [start + idx for idx in obs_indices], dtype=torch.int64
                )
            else:
                data = row[self._alias_to_native[key]]
            if isinstance(data, torch.Tensor):
                if data.ndim == 4 and data.shape[-1] in (1, 3):
                    data = data.permute(0, 3, 1, 2)
            steps[key] = data
        return self.transform(steps) if self.transform else steps


def _open_pair(root, **kwargs):
    return _open(root, **kwargs), _open_legacy(root, **kwargs)


def _open_legacy(root, **kwargs):
    kwargs.setdefault('video_backend', VIDEO_BACKEND)
    kwargs.setdefault('primary_camera_key', FRONT)
    kwargs.setdefault('key_aliases', KEY_ALIASES)
    return _LegacyAdapter(repo_id=REPO_ID, root=root, **kwargs)


def _chunk_specs(dataset) -> list[tuple[int, int, int]]:
    """(episode, start, end) slices, a whole number of frameskips long:
    one inside each episode and one that runs past its end."""
    fs = dataset.frameskip
    specs = []
    for ep, length in enumerate(dataset.lengths):
        length = int(length)
        specs.append((ep, 0, fs * max(1, length // fs)))
        start = max(0, length - fs - 1)
        specs.append((ep, start, start + 3 * fs))
    return specs


def _assert_same_windows(new, old) -> None:
    assert len(new) == len(old)
    for i in range(len(old)):
        assert_same(old[i], new[i], f'item {i}')
    for ep in range(len(old.lengths)):
        assert_same(old.load_episode(ep), new.load_episode(ep), f'ep {ep}')
    for ep, start, end in _chunk_specs(old):
        args = (np.array([ep]), np.array([start]), np.array([end]))
        assert_same(
            old.load_chunk(*args)[0],
            new.load_chunk(*args)[0],
            f'chunk {(ep, start, end)}',
        )
    # One-row slices, as GoalDataset reads its goals.
    for ep, length in enumerate(old.lengths):
        for step in (0, int(length) - 1):
            assert_same(
                old._load_slice(ep, step, step + 1),
                new._load_slice(ep, step, step + 1),
                f'row {(ep, step)}',
            )


@pytest.mark.parametrize('episodes', [None, [2, 0], [1]])
@pytest.mark.parametrize('num_steps', [1, 2, 4])
@pytest.mark.parametrize('frameskip', [1, 3])
def test_windows_match_the_old_code(
    multi_root, frameskip, num_steps, episodes
):
    new, old = _open_pair(
        multi_root, frameskip=frameskip, num_steps=num_steps, episodes=episodes
    )
    _assert_same_windows(new, old)


def test_image_primary_camera_matches_the_old_code(multi_root):
    for frameskip, num_steps in [(1, 1), (3, 2)]:
        new, old = _open_pair(
            multi_root,
            frameskip=frameskip,
            num_steps=num_steps,
            primary_camera_key=TOP,
            key_aliases={FRONT: 'front'},
        )
        assert new[0]['pixels'].shape[-3:] == (3, 16, 16)
        _assert_same_windows(new, old)


@pytest.mark.parametrize(
    'lerobot_kwargs',
    [{'return_uint8': True}, {'tolerance_s': 1e-3}],
    ids=['uint8', 'tolerance'],
)
def test_lerobot_options_match_the_old_code(multi_root, lerobot_kwargs):
    new, old = _open_pair(multi_root, num_steps=2, **lerobot_kwargs)
    _assert_same_windows(new, old)


def test_goal_dataset_matches_the_old_code(multi_root):
    new, old = _open_pair(multi_root, num_steps=2, frameskip=2)
    goals_new = GoalDataset(new, seed=0)
    goals_old = GoalDataset(old, seed=0)
    assert len(goals_new) == len(goals_old)
    for i in range(len(goals_old)):
        assert_same(goals_old[i], goals_new[i], f'goal item {i}')


def test_frames_come_from_the_right_row_and_camera(multi_root):
    """Checks every frame against the level it was written with.

    Episodes share one video file per camera, so this also checks the
    offset of each episode inside the file (``from_timestamp``).
    """
    # h264 moves a flat grey level by 2 at most; neighbouring rows differ
    # by 8 levels.
    atol = 3 / 255
    for episodes in (None, [2, 0]):
        adapter = _open(multi_root, episodes=episodes)
        absolute = sorted(episodes) if episodes else range(len(EP_LENGTHS))
        for local_ep, ep in enumerate(absolute):
            episode = adapter.load_episode(local_ep)
            for alias, key in (
                ('pixels', FRONT),
                ('wrist', WRIST),
                ('top', TOP),
            ):
                want = torch.tensor(
                    [level(key, ep, s) / 255 for s in range(EP_LENGTHS[ep])]
                )
                got = episode[alias].mean(dim=(1, 2, 3))
                tol = 1e-6 if key == TOP else atol  # PNG is lossless
                torch.testing.assert_close(got, want, atol=tol, rtol=0)


def _count_decodes(monkeypatch) -> dict[str, int]:
    """Count video decode calls per camera, at LeRobot's pyav backend."""
    from lerobot.datasets import video_utils

    calls: dict[str, int] = {}
    decode = video_utils.decode_video_frames_pyav

    def counting(video_path, *args, **kwargs):
        camera = next(k for k in CAMERAS if k in str(video_path))
        calls[camera] = calls.get(camera, 0) + 1
        return decode(video_path, *args, **kwargs)

    monkeypatch.setattr(video_utils, 'decode_video_frames_pyav', counting)
    return calls


def test_only_requested_cameras_are_decoded(multi_root, monkeypatch):
    adapter = _open(multi_root, num_steps=2, key_aliases={})
    calls = _count_decodes(monkeypatch)
    for i in range(5):
        adapter[i]
    adapter.load_episode(1)
    # One decode call per window for the requested camera, none for the
    # other one.
    assert calls == {FRONT: 6}


def test_one_lerobot_dataset_for_every_window_shape(multi_root, monkeypatch):
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    inits = []
    init = LeRobotDataset.__init__

    def counting(self, *args, **kwargs):
        inits.append(kwargs.get('delta_timestamps'))
        init(self, *args, **kwargs)

    monkeypatch.setattr(LeRobotDataset, '__init__', counting)
    adapter = _open(multi_root, num_steps=2, frameskip=2)
    adapter[0]
    for ep in range(len(EP_LENGTHS)):
        adapter.load_episode(ep)
    adapter.load_chunk(np.array([2]), np.array([10]), np.array([16]))
    GoalDataset(adapter, seed=0)[3]
    assert inits == [None]


# -- Depth cameras ------------------------------------------------------------

DEPTH_VIDEO = 'observation.images.depth'
DEPTH_IMAGE = 'observation.images.depth_png'


def _depth_m(key: str, ep: int, step: int) -> float:
    return 0.5 + 0.1 * (EP_OFFSETS[ep] + step)


@pytest.fixture(scope='module')
def depth_root(tmp_path_factory):
    """Two depth cameras: a depth video and a depth image."""
    import av

    try:
        av.codec.Codec('hevc', 'w')
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f'no hevc encoder for depth videos ({exc})')
    return write_dataset(
        tmp_path_factory.mktemp('lerobot') / 'ds',
        repo_id=REPO_ID,
        lengths=EP_LENGTHS,
        cameras={FRONT: 'video'},
        level=level,
        depth_cameras={DEPTH_VIDEO: 'video', DEPTH_IMAGE: 'image'},
        depth_m=_depth_m,
    )


@pytest.mark.parametrize('depth_output_unit', ['mm', 'm'])
def test_depth_cameras_match_the_old_code(depth_root, depth_output_unit):
    new, old = _open_pair(
        depth_root,
        num_steps=2,
        frameskip=3,
        key_aliases={DEPTH_VIDEO: 'depth', DEPTH_IMAGE: 'depth_png'},
        depth_output_unit=depth_output_unit,
    )
    _assert_same_windows(new, old)

    # The depth values are right, in the output unit.
    scale = 1000.0 if depth_output_unit == 'mm' else 1.0
    episode = new.load_episode(1)
    want = torch.tensor([_depth_m(DEPTH_VIDEO, 1, s) for s in range(0, 9, 3)])
    for alias in ('depth', 'depth_png'):
        assert episode[alias].shape == (3, 1, 16, 16)
        torch.testing.assert_close(
            episode[alias].mean(dim=(1, 2, 3)), scale * want, rtol=2e-3, atol=0
        )
