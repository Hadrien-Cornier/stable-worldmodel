"""Tiny LeRobot v3 datasets for the LeRobot adapter tests.

Test modules in this directory import this file directly (``from
_lerobot_data import ...``), like ``_guards.py``. Nothing is downloaded:
:func:`offline_hf` switches Hub access off and keeps the ``datasets`` cache
in a temp folder.

Every row stores where it comes from, so tests can check that the adapter
returns the right rows and frames, not only the right shapes:

  - ``observation.state`` is ``[episode, step]``
  - ``action`` is ``[episode, step + 0.5]``
  - ``next.reward`` is ``episode + step / 100``, and ``next.done`` is true
    on the last step of each episode
  - every camera frame is one flat grey level, given by the ``level``
    function passed to :func:`write_dataset`
"""

from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping, Sequence
from pathlib import Path

import numpy as np
import pytest

FPS = 10
HW = 16  # h264 with yuv420p needs even frame sizes.
ACTION_DIM = 2

# torchcodec needs FFmpeg shared libraries that CI runners may not have.
# PyAV ships its own FFmpeg, so decode with it.
VIDEO_BACKEND = 'pyav'


def offline_hf(tmp_path_factory) -> Iterator[None]:
    """Keep every Hugging Face read offline and inside a temp folder.

    Use it as the body of a module-scoped autouse fixture.
    """
    import datasets
    import huggingface_hub.constants

    cache = tmp_path_factory.mktemp('hf_datasets_cache')
    with pytest.MonkeyPatch.context() as mp:
        mp.setenv('HF_HUB_OFFLINE', '1')
        mp.setenv('HF_DATASETS_CACHE', str(cache))
        # Both libraries read these variables once, at import time.
        mp.setattr(huggingface_hub.constants, 'HF_HUB_OFFLINE', True)
        mp.setattr(datasets.config, 'HF_HUB_OFFLINE', True)
        mp.setattr(datasets.config, 'HF_DATASETS_CACHE', cache)
        yield


def action(ep: int, step: int) -> list[float]:
    return [float(ep), step + 0.5]


def state(ep: int, step: int) -> list[float]:
    return [float(ep), float(step)]


def write_dataset(
    root: Path,
    *,
    repo_id: str,
    lengths: Sequence[int],
    cameras: Mapping[str, str],
    level: Callable[[str, int, int], int],
) -> Path:
    """Write a LeRobot v3 dataset to ``root`` and return ``root``.

    Args:
        root: Folder to create.
        repo_id: Repo id stored in the metadata (never contacted).
        lengths: Number of steps of each episode.
        cameras: Camera key to storage, ``'video'`` (h264) or ``'image'``
            (PNG inside the parquet files).
        level: ``level(camera_key, episode, step)`` is the grey level
            (0-255) of that frame.
    """
    from lerobot.configs.video import RGBEncoderConfig
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    features = {
        key: {
            'dtype': storage,
            'shape': (HW, HW, 3),
            'names': ['height', 'width', 'channels'],
        }
        for key, storage in cameras.items()
    }
    features.update(
        {
            'observation.state': {
                'dtype': 'float32',
                'shape': (2,),
                'names': ['ep', 'step'],
            },
            'action': {
                'dtype': 'float32',
                'shape': (ACTION_DIM,),
                'names': ['ep', 'step'],
            },
            'next.reward': {'dtype': 'float32', 'shape': (1,), 'names': None},
            'next.done': {'dtype': 'bool', 'shape': (1,), 'names': None},
        }
    )
    writer = LeRobotDataset.create(
        repo_id=repo_id,
        fps=FPS,
        features=features,
        root=root,
        use_videos=True,
        rgb_encoder=RGBEncoderConfig(vcodec='h264'),
    )
    for ep, length in enumerate(lengths):
        for step in range(length):
            frame = {
                key: np.full((HW, HW, 3), level(key, ep, step), dtype=np.uint8)
                for key in cameras
            }
            frame.update(
                {
                    'observation.state': np.array(
                        state(ep, step), dtype=np.float32
                    ),
                    'action': np.array(action(ep, step), dtype=np.float32),
                    'next.reward': np.array(
                        [ep + step / 100], dtype=np.float32
                    ),
                    'next.done': np.array([step == length - 1]),
                    'task': 'tiny',
                }
            )
            writer.add_frame(frame)
        writer.save_episode(parallel_encoding=False)
    writer.finalize()
    return root
