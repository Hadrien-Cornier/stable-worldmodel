"""LeRobot Hub format (read-only).

Identified by the ``lerobot://`` scheme. Mapping ``World.collect``'s
arbitrary info-dict to LeRobot's prescribed schema is non-trivial and
therefore not supported as a writer here.

The adapter keeps one ``LeRobotDataset``, for download, metadata, the
episode filter and ``hf_dataset``. It reads each window itself: table
columns straight from Arrow, and video frames with LeRobot's
``decode_video_frames``, only for the cameras in ``keys_to_load``.
"""

from __future__ import annotations

import sys
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any, NamedTuple

import numpy as np
import torch

from stable_worldmodel.data.dataset import Dataset
from stable_worldmodel.data.format import Format, register_format

_SCHEME = 'lerobot://'


def _import_lerobot_hub_dataset() -> type:
    """Import upstream lerobot `LeRobotDataset` lazily (aliased to avoid name clash)."""
    if sys.version_info < (3, 12):
        raise ImportError(
            'stable_worldmodel.data.LeRobotAdapter requires Python 3.12+ because '
            'the official lerobot package is only available on Python 3.12+.'
        )

    try:
        from lerobot.datasets.lerobot_dataset import (
            LeRobotDataset as LerobotHubDataset,
        )
    except ImportError as exc:
        raise ImportError(
            'stable_worldmodel.data.LeRobotAdapter requires the optional '
            'lerobot dependency. Install it with '
            "`pip install 'stable-worldmodel[lerobot]'`. "
            f'Underlying error: {exc}'
        ) from exc

    return LerobotHubDataset


class _EpisodeStructure(NamedTuple):
    """Where each episode sits in the flat rows of a LeRobot dataset."""

    local_episode_index: np.ndarray  # per row: episode number 0..E-1
    step_idx: np.ndarray  # per row: step inside its episode
    lengths: np.ndarray  # per episode: number of rows
    offsets: np.ndarray  # per episode: first row
    absolute_episode_ids: np.ndarray  # per episode: LeRobot episode_index


def _episode_structure(episode_index: np.ndarray) -> _EpisodeStructure:
    """Find the episodes in LeRobot's per-row ``episode_index`` column.

    LeRobot stores the rows of one episode next to each other, so each
    episode is one run of equal values. Episodes are numbered in the order
    of their first row. Every step is a numpy operation, so a million rows
    take milliseconds.

    Raises:
        ValueError: If the rows of one episode are not contiguous.
    """
    ids = np.asarray(episode_index).astype(np.int64, copy=False)
    n_rows = len(ids)
    if n_rows == 0:
        offsets = np.zeros(0, dtype=np.int64)
    else:
        run_starts = np.flatnonzero(ids[1:] != ids[:-1]) + 1
        offsets = np.concatenate(([0], run_starts)).astype(np.int64)
    lengths = np.diff(np.append(offsets, n_rows)).astype(np.int64)
    absolute_episode_ids = ids[offsets]

    unique_ids, counts = np.unique(absolute_episode_ids, return_counts=True)
    if (counts > 1).any():
        split = int(unique_ids[counts > 1][0])
        raise ValueError(
            f'The rows of LeRobot episode {split} are not contiguous. '
            'LeRobotAdapter needs the rows of each episode next to each '
            'other.'
        )

    local_episode_index = np.repeat(
        np.arange(len(offsets), dtype=np.int64), lengths
    )
    step_idx = np.arange(n_rows, dtype=np.int64) - np.repeat(offsets, lengths)
    return _EpisodeStructure(
        local_episode_index, step_idx, lengths, offsets, absolute_episode_ids
    )


def _numpy_layout(arrow_type: Any) -> tuple[np.dtype, tuple[int, ...]] | None:
    """Numpy dtype and per-row shape for a column of this Arrow type.

    The dtypes are the ones LeRobot's ``hf_transform_to_torch`` gives, since
    it calls ``torch.tensor`` on Python values: floats become float32,
    integers int64, and booleans stay bool. Each fixed-size list adds one
    dimension. Any other type (images, strings, variable-length lists,
    extension types) returns ``None``.
    """
    import pyarrow as pa

    row_shape = []
    while pa.types.is_fixed_size_list(arrow_type):
        row_shape.append(arrow_type.list_size)
        arrow_type = arrow_type.value_type
    if pa.types.is_floating(arrow_type):
        dtype = np.float32
    elif pa.types.is_integer(arrow_type) and arrow_type != pa.uint64():
        dtype = np.int64
    elif pa.types.is_boolean(arrow_type):
        dtype = np.bool_
    else:
        return None
    return np.dtype(dtype), tuple(row_shape)


def _arrow_to_numpy(column: Any) -> np.ndarray | None:
    """Copy an Arrow array or chunked array into a new numpy array.

    Returns ``None`` when the type has no numpy layout (see
    :func:`_numpy_layout`) or the column holds nulls. The caller then reads
    the column through ``datasets`` instead.
    """
    import pyarrow as pa

    layout = _numpy_layout(column.type)
    if layout is None:
        return None
    dtype, row_shape = layout
    if isinstance(column, pa.ChunkedArray):
        column = column.combine_chunks()
    values = column
    for _ in row_shape:
        if values.null_count:
            return None
        values = values.flatten()
    if values.null_count:
        return None
    flat = np.array(values.to_numpy(zero_copy_only=False), dtype=dtype)
    return flat.reshape((len(column), *row_shape))


def _lerobot_video_utils() -> Any:
    """LeRobot's video module, imported on first use.

    Callers look up ``decode_video_frames`` on it at call time, so a patched
    or wrapped decoder is picked up.
    """
    from lerobot.datasets import video_utils

    return video_utils


def _scalarize(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        if value.ndim == 0:
            return value.item()
        return value.detach().cpu().numpy()
    if isinstance(value, np.ndarray):
        if value.ndim == 0:
            return value.item()
        return value
    return value


def _column_to_numpy(column: Any) -> np.ndarray:
    if isinstance(column, torch.Tensor):
        return column.detach().cpu().numpy()
    if isinstance(column, np.ndarray):
        return column
    if isinstance(column, list):
        return np.asarray([_scalarize(v) for v in column])
    return np.asarray(column)


class LeRobotAdapter(Dataset):
    """Wraps lerobot's `LeRobotDataset` and exposes the SWM `Dataset` API."""

    _SYNTHETIC_COLUMNS = {'ep_idx', 'step_idx'}

    def __init__(
        self,
        repo_id: str,
        root: str | Path | None = None,
        episodes: list[int] | None = None,
        frameskip: int = 1,
        num_steps: int = 1,
        transform: Callable[[dict], dict] | None = None,
        keys_to_load: list[str] | None = None,
        keys_to_cache: list[str] | None = None,
        primary_camera_key: str | None = None,
        key_aliases: dict[str, str] | None = None,
        **lerobot_kwargs: Any,
    ) -> None:
        LerobotHubDataset = _import_lerobot_hub_dataset()
        self._lerobot_kwargs = dict(lerobot_kwargs)
        self.dataset = LerobotHubDataset(
            repo_id=repo_id,
            root=root,
            episodes=episodes,
            image_transforms=None,
            delta_timestamps=None,
            **lerobot_kwargs,
        )
        self.repo_id = repo_id
        self.root = Path(root) if root is not None else None
        self.episodes = episodes

        native_keys = self._get_native_keys()
        self._camera_keys = self.dataset.meta.camera_keys
        self._fps = self._get_fps()
        self._primary_camera_key = self._resolve_primary_camera(
            primary_camera_key, native_keys
        )

        self._native_to_alias = self._build_alias_map(native_keys, key_aliases)
        self._alias_to_native = {
            alias: native for native, alias in self._native_to_alias.items()
        }
        self._full_columns: dict[str, np.ndarray] = {}
        self._arrow_columns: dict[str, Any] = {}
        self._views: dict[str, Any] = {}
        self._video_sources: dict[tuple[str, int], tuple[Path, float]] = {}
        self._cache: dict[str, np.ndarray] = {}
        self._setup_video_decoding()

        structure = _episode_structure(
            self._get_native_column('episode_index')
        )
        self._absolute_episode_ids = structure.absolute_episode_ids
        self._cache['ep_idx'] = structure.local_episode_index
        self._cache['step_idx'] = structure.step_idx

        if keys_to_load is None:
            keys_to_load = list(self._native_to_alias.values()) + [
                'ep_idx',
                'step_idx',
            ]
        self._keys = list(dict.fromkeys(keys_to_load))

        for key in keys_to_cache or []:
            self._cache[key] = self._materialize_column(key)

        super().__init__(
            structure.lengths,
            structure.offsets,
            frameskip,
            num_steps,
            transform,
        )

    @property
    def column_names(self) -> list[str]:
        return self._keys

    #: Caches that are cheap to rebuild. They are left out of the pickle
    #: that DataLoader workers receive, and each worker refills them on use.
    _UNPICKLED_CACHES = (
        '_full_columns',
        '_arrow_columns',
        '_views',
        '_video_sources',
    )

    def __getstate__(self) -> dict:
        state = self.__dict__.copy()
        for name in self._UNPICKLED_CACHES:
            state[name] = {}
        # A trainer attached as `dataset._trainer` (stable-pretraining's
        # pattern) reaches the DataLoader iterator, which cannot be
        # pickled. Leave it out, as LanceDataset does.
        state['_trainer'] = None
        return state

    def _setup_video_decoding(self) -> None:
        """Read the decode settings the way ``LeRobotDataset`` does."""
        from lerobot.utils.import_utils import get_safe_default_video_backend

        meta = self.dataset.meta
        self._video_keys = frozenset(meta.video_keys)
        self._video_backend = (
            self._lerobot_kwargs.get('video_backend')
            or get_safe_default_video_backend()
        )
        self._return_uint8 = bool(
            self._lerobot_kwargs.get('return_uint8', False)
        )
        self._depth_output_unit = self.dataset.depth_output_unit

        # Depth videos are dequantized after decoding. Depth images are
        # converted to the output unit when they were stored in another one.
        self._depth_encoders: dict[str, Any] = {}
        depth_videos = [k for k in meta.depth_keys if k in self._video_keys]
        if depth_videos:
            from lerobot.configs import DepthEncoderConfig

            self._depth_encoders = {
                key: DepthEncoderConfig.from_video_info(
                    meta.features[key].get('info')
                )
                for key in depth_videos
            }
        self._image_depth_units = {
            key: (meta.features[key].get('info') or {}).get('depth_unit')
            for key in meta.depth_keys
            if key in meta.image_keys
        }

    def _get_native_keys(self) -> list[str]:
        features = self.dataset.features
        if not isinstance(features, Mapping):
            raise TypeError(
                'LeRobot dataset features must be a mapping of column names.'
            )
        return list(features.keys())

    def _resolve_primary_camera(
        self,
        primary_camera_key: str | None,
        native_keys: list[str],
    ) -> str | None:
        if primary_camera_key is None and len(self._camera_keys) > 1:
            raise ValueError(
                'LeRobotAdapter requires `primary_camera_key` when '
                'multiple cameras are available.'
            )
        if primary_camera_key is not None:
            if primary_camera_key not in native_keys:
                raise KeyError(
                    f"Primary camera key '{primary_camera_key}' not found in LeRobot dataset."
                )
            return primary_camera_key

        for key in self._camera_keys:
            if key in native_keys:
                return key
        return None

    def _get_fps(self) -> float:
        meta = self.dataset.meta
        info = meta.info

        if isinstance(info, dict) and 'fps' in info:
            return float(info['fps'])
        if hasattr(info, 'fps'):
            return float(info.fps)

        raise ValueError(
            'LeRobot dataset metadata must expose `meta.info.fps`.'
        )

    def _build_alias_map(
        self,
        native_keys: list[str],
        key_aliases: dict[str, str] | None,
    ) -> dict[str, str]:
        aliases: dict[str, str] = {}
        if self._primary_camera_key is not None:
            aliases[self._primary_camera_key] = 'pixels'
        if 'action' in native_keys:
            aliases['action'] = 'action'
        if 'observation.state' in native_keys:
            aliases['observation.state'] = 'proprio'

        for native, alias in (key_aliases or {}).items():
            if native not in native_keys:
                raise KeyError(
                    f"Key alias source '{native}' not found in LeRobot dataset."
                )
            aliases[native] = alias

        return aliases

    def _get_native_column(self, native_key: str) -> np.ndarray:
        if native_key not in self._full_columns:
            values = None
            column = self._arrow_column(native_key)
            if column is not None:
                values = _arrow_to_numpy(column)
            if values is None:
                # Row by row through `datasets`, as before this change.
                values = _column_to_numpy(
                    self._column_view(native_key)[native_key]
                )
            self._full_columns[native_key] = values
        return self._full_columns[native_key]

    def _arrow_column(self, native_key: str) -> Any:
        """The Arrow data of a table column, or ``None``.

        ``None`` means the column type has no numpy layout (see
        :func:`_numpy_layout`), or ``hf_dataset`` has an indices mapping
        (from ``select`` or ``shuffle``) that the Arrow table does not
        follow. LeRobot does not use one.
        """
        if native_key not in self._arrow_columns:
            hf_dataset = self.dataset.hf_dataset
            column = None
            if getattr(hf_dataset, '_indices', None) is None:
                column = hf_dataset.data.column(native_key)
                if _numpy_layout(column.type) is None:
                    column = None
            self._arrow_columns[native_key] = column
        return self._arrow_columns[native_key]

    def _column_view(self, native_key: str) -> Any:
        """One-column view of ``hf_dataset``, with LeRobot's transform.

        ``select_columns`` does not copy data. Reading rows of the view
        decodes only this column, while ``hf_dataset[rows]`` decodes every
        column, including every image.
        """
        if native_key not in self._views:
            self._views[native_key] = self.dataset.hf_dataset.select_columns(
                [native_key]
            )
        return self._views[native_key]

    def _materialize_column(self, key: str) -> np.ndarray:
        if key in self._cache:
            return self._cache[key]
        if key in self._SYNTHETIC_COLUMNS:
            return self._cache[key]

        native_key = self._alias_to_native.get(key)
        if native_key is None:
            raise KeyError(f"Unknown LeRobot adapter column '{key}'.")
        if native_key in self._camera_keys:
            raise KeyError(
                f"'{key}' cannot be materialized as a full array because it is image/video-backed."
            )
        return self._get_native_column(native_key)

    def _window_rows(self, key: str, length: int) -> np.ndarray:
        """Offsets, from the window start, of the rows that ``key`` reads.

        ``action`` reads every row, so the actions between two observations
        are kept. Every other column reads one row every ``frameskip`` rows.
        """
        step = 1 if key == 'action' else self.frameskip
        return np.arange(0, length, step, dtype=np.int64)

    def _load_slice(self, ep_idx: int, start: int, end: int) -> dict:
        g_start = int(self.offsets[ep_idx] + start)
        length = int(end - start)
        # As in LeRobot, a window that runs past the end of its episode
        # repeats the last row. The episode is the one that holds g_start.
        row_ep = int(self._cache['ep_idx'][g_start])
        first_row = int(self.offsets[row_ep])
        last_row = first_row + int(self.lengths[row_ep]) - 1
        steps: dict[str, Any] = {}
        for key in self._keys:
            offsets = self._window_rows(key, length)
            if key == 'ep_idx':
                data = torch.full((len(offsets),), row_ep, dtype=torch.int64)
            elif key == 'step_idx':
                # Not clamped: past the end of the episode it keeps counting.
                data = torch.as_tensor(start + offsets, dtype=torch.int64)
            else:
                rows = np.clip(g_start + offsets, first_row, last_row)
                data = self._read_window(
                    self._alias_to_native[key], rows, row_ep
                )

            if isinstance(data, torch.Tensor):
                if data.ndim == 4 and data.shape[-1] in (1, 3):
                    data = data.permute(0, 3, 1, 2)
            steps[key] = data

        return self.transform(steps) if self.transform else steps

    def _read_window(
        self, native_key: str, rows: np.ndarray, row_ep: int
    ) -> torch.Tensor:
        """The values of ``native_key`` at ``rows``, stacked on a new axis."""
        if native_key in self._video_keys:
            return self._decode_video(native_key, rows, row_ep)

        data = self._read_rows(native_key, rows)
        stored_unit = self._image_depth_units.get(native_key)
        if stored_unit is not None and stored_unit != self._depth_output_unit:
            from lerobot.configs import DEPTH_METER_UNIT
            from lerobot.datasets.depth_utils import MM_PER_METRE

            if stored_unit == DEPTH_METER_UNIT:
                data = data * MM_PER_METRE
            else:
                data = data / MM_PER_METRE
        return data

    def _read_rows(self, native_key: str, rows: np.ndarray) -> torch.Tensor:
        column = self._arrow_column(native_key)
        if column is not None:
            lo, hi = int(rows.min()), int(rows.max()) + 1
            values = _arrow_to_numpy(column.slice(lo, hi - lo))
            if values is not None:
                return torch.from_numpy(values[rows - lo])
        # Images and other types go through LeRobot's transform, on a
        # one-column view so no other column is decoded.
        view = self._column_view(native_key)
        return torch.stack(view[rows.tolist()][native_key])

    def _decode_video(
        self, native_key: str, rows: np.ndarray, row_ep: int
    ) -> torch.Tensor:
        """Decode one camera's frames at ``rows``, as LeRobot does."""
        video_path, from_timestamp = self._video_source(
            native_key, int(self._absolute_episode_ids[row_ep])
        )
        # The same float64 sums as LeRobot: where the episode starts in the
        # video file, plus the timestamp of each row.
        timestamps = self._read_rows('timestamp', rows).numpy()
        query = (from_timestamp + timestamps.astype(np.float64)).tolist()
        frames = _lerobot_video_utils().decode_video_frames(
            video_path,
            query,
            self.dataset.tolerance_s,
            self._video_backend,
            return_uint8=self._return_uint8,
            is_depth=native_key in self._depth_encoders,
        )
        depth = self._depth_encoders.get(native_key)
        if depth is not None:
            from lerobot.datasets.depth_utils import dequantize_depth

            frames = dequantize_depth(
                frames,
                depth_min=depth.depth_min,
                depth_max=depth.depth_max,
                shift=depth.shift,
                use_log=depth.use_log,
                output_unit=self._depth_output_unit,
            )
        # LeRobot drops the time axis of a one-frame window, so a video
        # camera gives (C, H, W) there. Kept as it was.
        return frames.squeeze(0)

    def _video_source(
        self, native_key: str, absolute_episode: int
    ) -> tuple[Path, float]:
        """Video file of one camera and episode, and the episode's start."""
        cache_key = (native_key, absolute_episode)
        if cache_key not in self._video_sources:
            meta = self.dataset.meta
            video_path = self.dataset.root / meta.get_video_file_path(
                absolute_episode, native_key
            )
            episode = meta.episodes[absolute_episode]
            from_timestamp = episode[f'videos/{native_key}/from_timestamp']
            self._video_sources[cache_key] = (video_path, from_timestamp)
        return self._video_sources[cache_key]

    def get_col_data(self, col: str) -> np.ndarray:
        return self._materialize_column(col)

    def get_row_data(self, row_idx: int | list[int]) -> dict:
        out = {}
        for col in self._keys:
            try:
                data = self._materialize_column(col)
            except KeyError:
                continue
            out[col] = data[row_idx]
        return out

    def get_dim(self, col: str) -> int:
        data = self.get_col_data(col)
        return np.prod(data.shape[1:]).item() if data.ndim > 1 else 1


@register_format
class LeRobot(Format):
    name = 'lerobot'

    @classmethod
    def detect(cls, path) -> bool:
        return isinstance(path, str) and path.startswith(_SCHEME)

    @classmethod
    def open_reader(cls, path, **kwargs):
        repo_id = path[len(_SCHEME) :] if path.startswith(_SCHEME) else path
        return LeRobotAdapter(repo_id, **kwargs)


__all__ = ['LeRobot', 'LeRobotAdapter']
