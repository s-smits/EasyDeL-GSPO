# Copyright 2025 The EasyDeL Author @erfanzar (Erfan Zare Chavoshi).
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

import os
import re
import typing as tp

try:
    import jax
except Exception:  # pragma: no cover - runtime import guard
    jax = None  # type: ignore

from easydel.utils.helpers import get_logger

logger = get_logger(__name__)


def _env_bool(name: str, default: bool) -> bool:
    val = os.environ.get(name)
    if val is None:
        return default
    return str(val).strip().lower() in {"1", "true", "yes", "y", "on"}


def _env_int(name: str, default: int | None) -> int | None:
    val = os.environ.get(name)
    if val is None or str(val).strip() == "":
        return default
    try:
        return int(val)
    except Exception:
        return default


def _process_index() -> int:
    try:
        if jax is not None:
            return int(jax.process_index())  # type: ignore[attr-defined]
    except Exception:
        pass
    return int(os.environ.get("JAX_PROCESS_INDEX", "0") or 0)


def is_admin_host() -> bool:
    only_zero = _env_bool("ED_HF_ONLY_PROCESS_ZERO", True)
    if not only_zero:
        return True
    return _process_index() == 0


def should_upload_enabled() -> bool:
    # Primary gate; default disabled unless explicitly enabled
    if not _env_bool("ED_HF_UPLOAD", False) and not _env_bool("HF_UPLOAD", False):
        return False
    return is_admin_host()


def _get_token(passed: str | None = None) -> str | None:
    if passed:
        return passed
    # Respect both HF_TOKEN and HUGGINGFACE_TOKEN
    return os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")


def _natural_run_key(path: str) -> tuple[int, str]:
    # Extract run number from directories named 'run-<int>' (defaults to -1)
    m = re.match(r"^run-(\d+)$", path)
    if not m:
        return (-1, path)
    try:
        return (int(m.group(1)), path)
    except Exception:
        return (-1, path)


def prune_remote_checkpoints(repo_id: str, *, keep_n: int, branch: str | None = None, token: str | None = None) -> None:
    """Keep only the latest N 'run-*' directories at repo root on HF Hub.

    This function lists repo tree, identifies top-level 'run-*' directories,
    and deletes older ones beyond keep_n.
    """
    try:
        from huggingface_hub import HfApi
    except Exception:  # pragma: no cover - optional dependency
        logger.warning("huggingface_hub not available; skipping remote pruning")
        return

    api = HfApi(token=_get_token(token))
    try:
        items = api.list_repo_tree(repo_id=repo_id, repo_type="model", revision=branch or "main", recursive=True)
    except Exception as e:
        logger.warning(f"Failed to list repo tree for pruning: {e}")
        return

    # Identify top-level run-* directories
    top_level_run_dirs: set[str] = set()
    for entry in items:
        path = getattr(entry, "path", "")
        # only top-level directories (no '/')
        if "/" in path:
            continue
        if getattr(entry, "type", "") == "directory" and path.startswith("run-"):
            top_level_run_dirs.add(path)

    if len(top_level_run_dirs) <= keep_n:
        return

    # Sort by run number (fallback alphabetical)
    sorted_dirs = sorted(top_level_run_dirs, key=_natural_run_key)
    to_delete = sorted_dirs[: max(0, len(sorted_dirs) - int(keep_n))]
    if not to_delete:
        return

    # Build a set of all paths to delete (folder contents)
    delete_paths: list[str] = []
    for entry in items:
        ep = getattr(entry, "path", "")
        if any(ep == d or ep.startswith(d + "/") for d in to_delete):
            delete_paths.append(ep)

    # Delete collected paths (files), skipping directories
    for p in delete_paths:
        try:
            if p.endswith("/"):
                continue
            api.delete_file(path=p, repo_id=repo_id, repo_type="model")
        except Exception:
            # Best-effort; continue on delete failures
            pass


def upload_folder_with_keep_n(
    *,
    folder_path: str,
    repo_id: str,
    token: str | None = None,
    private: bool | None = None,
    branch: str | None = None,
    commit_message: str | None = None,
    keep_n: int | None = None,
) -> str | None:
    """Upload a local folder to HF Hub and optionally prune older run-* directories.

    Returns the URL of the repo HEAD after upload if available.
    """
    try:
        from huggingface_hub import HfApi
    except Exception:  # pragma: no cover
        logger.warning("huggingface_hub not available; skipping upload")
        return None

    api = HfApi(token=_get_token(token))
    try:
        api.create_repo(repo_id=repo_id, repo_type="model", private=bool(private), exist_ok=True)
    except Exception as e:
        logger.warning(f"Failed to create or access repo {repo_id}: {e}")

    try:
        api.upload_folder(
            folder_path=folder_path,
            repo_id=repo_id,
            repo_type="model",
            revision=branch,
            commit_message=commit_message or f"upload {os.path.basename(folder_path)}",
        )
    except Exception as e:
        logger.warning(f"HF upload_folder failed: {e}")
        return None

    if keep_n is not None and keep_n >= 0:
        try:
            prune_remote_checkpoints(repo_id=repo_id, keep_n=int(keep_n), branch=branch, token=token)
        except Exception:
            pass

    try:
        return f"https://huggingface.co/{repo_id}"
    except Exception:
        return None


def upload_checkpoint_dir_if_enabled(local_dir: str, *, repo_id: str | None = None) -> None:
    """Upload a checkpoint directory to HF Hub if env gates permit.

    Env gates:
      - ED_HF_UPLOAD/HF_UPLOAD: enable uploads (default off)
      - ED_HF_ONLY_PROCESS_ZERO: only process 0 performs uploads (default on)
      - ED_HF_REPO_ID: default repo id when not provided
      - ED_HF_PRIVATE: upload to private repo (default True)
      - ED_HF_KEEP_N: keep at most N 'run-*' directories (default None)
      - HF_TOKEN/HUGGINGFACE_TOKEN: authentication token
    """
    if not should_upload_enabled():
        return

    rid = repo_id or os.environ.get("ED_HF_REPO_ID") or os.path.basename(os.path.dirname(local_dir))
    private = _env_bool("ED_HF_PRIVATE", True)
    keep_n = _env_int("ED_HF_KEEP_N", None)
    branch = os.environ.get("ED_HF_BRANCH")
    token = _get_token(None)

    try:
        url = upload_folder_with_keep_n(
            folder_path=local_dir,
            repo_id=rid,
            token=token,
            private=private,
            branch=branch,
            keep_n=keep_n,
            commit_message=f"checkpoint: {os.path.basename(local_dir)}",
        )
        if url:
            logger.info(f"Uploaded checkpoint {local_dir} -> {url}")
    except Exception as e:
        logger.warning(f"HF upload skipped due to error: {e}")



