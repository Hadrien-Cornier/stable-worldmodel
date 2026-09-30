"""LeRobot Hub format (read-only).

Identified by the ``lerobot://`` scheme. Mapping ``World.collect``'s
arbitrary info-dict to LeRobot's prescribed schema is non-trivial and
therefore not supported as a writer here.
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
        self._hub_dataset_cls = LerobotHubDataset
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
        self._cache: dict[str, np.ndarray] = {}
        self._window_datasets: dict[
            tuple[tuple[int, ...], tuple[int, ...]], Any
        ] = {}

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
    _UNPICKLED_CACHES = ('_full_columns', '_arrow_columns', '_views')

    def __getstate__(self) -> dict:
        state = self.__dict__.copy()
        for name in self._UNPICKLED_CACHES:
            state[name] = {}
        # A trainer attached as `dataset._trainer` (stable-pretraining's
        # pattern) reaches the DataLoader iterator, which cannot be
        # pickled. Leave it out, as LanceDataset does.
        state['_trainer'] = None
        return state

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

    def _time_offsets(self, indices: tuple[int, ...]) -> list[float]:
        return [float(idx) / self._fps for idx in indices]

    def _window_dataset(
        self,
        observation_indices: tuple[int, ...],
        action_indices: tuple[int, ...],
    ) -> Any:
        cache_key = (observation_indices, action_indices)
        if cache_key not in self._window_datasets:
            delta_timestamps = {}
            for key in self._keys:
                if key in self._SYNTHETIC_COLUMNS:
                    continue
                native_key = self._alias_to_native.get(key)
                if native_key is None:
                    continue
                if key == 'action':
                    delta_timestamps[native_key] = self._time_offsets(
                        action_indices
                    )
                else:
                    delta_timestamps[native_key] = self._time_offsets(
                        observation_indices
                    )

            self._window_datasets[cache_key] = self._hub_dataset_cls(
                repo_id=self.repo_id,
                root=self.root,
                episodes=self.episodes,
                image_transforms=None,
                delta_timestamps=delta_timestamps or None,
                **self._lerobot_kwargs,
            )
        return self._window_datasets[cache_key]

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

    def _get_item_value(self, item: dict[str, Any], key: str) -> Any:
        if key == 'ep_idx':
            return int(self._cache['ep_idx'][item['_row_idx']])
        if key == 'step_idx':
            return int(self._cache['step_idx'][item['_row_idx']])

        native_key = self._alias_to_native[key]
        return item[native_key]

    def _load_slice(self, ep_idx: int, start: int, end: int) -> dict:
        g_start = int(self.offsets[ep_idx] + start)
        length = int(end - start)
        obs_indices = tuple(range(0, length, self.frameskip))
        action_indices = tuple(range(length))
        row = dict(self._window_dataset(obs_indices, action_indices)[g_start])
        row['_row_idx'] = g_start
        steps: dict[str, Any] = {}
        for key in self._keys:
            if key in self._SYNTHETIC_COLUMNS:
                if key == 'ep_idx':
                    data = torch.full(
                        (len(obs_indices),),
                        int(self._cache['ep_idx'][g_start]),
                        dtype=torch.int64,
                    )
                else:
                    data = torch.as_tensor(
                        [start + idx for idx in obs_indices],
                        dtype=torch.int64,
                    )
            else:
                data = self._get_item_value(row, key)

            if isinstance(data, torch.Tensor):
                if data.ndim == 4 and data.shape[-1] in (1, 3):
                    data = data.permute(0, 3, 1, 2)
            steps[key] = data

        return self.transform(steps) if self.transform else steps

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
