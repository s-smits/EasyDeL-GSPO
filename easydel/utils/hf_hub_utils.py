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
import threading
import typing as tp

from easydel.utils import EasyPath, EasyPathLike


def get_hf_token(env_name: str = "HF_TOKEN") -> str | None:
    """Return HF token from environment.

    Primary source is the provided env_name. Fallbacks: HF_TOKEN, HF_TOKEN_FOR_EASYDEL.
    """
    tok = os.getenv(env_name)
    if tok is None:
        tok = os.getenv("HF_TOKEN") or os.getenv("HF_TOKEN_FOR_EASYDEL")
    return tok


def should_push_to_hub(push_flag: bool | None, token: str | None) -> bool:
    """Determine whether checkpoint upload should occur.

    If push_flag is None, auto-enable when token is present.
    If push_flag is True, require token to be present.
    """
    if push_flag is not None:
        return bool(push_flag) and (token is not None)
    return token is not None


def resolve_repo_id(repo_id: str | None, model_name: str | None) -> str:
    base = repo_id or (model_name or "easydel-model")
    return str(base)


def upload_checkpoint_folder(
    *,
    local_ckpt_dir: EasyPathLike,
    repo_id: str,
    token: str,
    private: bool = True,
    path_in_repo_prefix: str = "checkpoints",
    keep_n: int | None = None,
) -> None:
    """Upload a local checkpoint directory to HF Hub under the given prefix.

    Best-effort and non-blocking by default:
    - Set EASYDEL_HF_SYNC=1 to force synchronous upload
    - Set EASYDEL_HF_TIMEOUT=<sec> to bound initial join wait (default 2s)
    - Set EASYDEL_DISABLE_HF_UPLOADS=1 to skip entirely
    """

    # Single consolid   ted gate to disable all uploads quickly
    if os.getenv("EASYDEL_DISABLE_HF_UPLOADS") in {"1", "true", "True"}:
        return

    def _worker():
        try:
            from huggingface_hub import HfApi
        except Exception:
            return

        api = HfApi()
        try:
            api.create_repo(repo_id=repo_id, repo_type="model", private=private, exist_ok=True, token=token)
        except Exception:
            ...

        ckpt_name = EasyPath(local_ckpt_dir).name
        path_in_repo = f"{path_in_repo_prefix}/{ckpt_name}"
        try:
            api.upload_folder(
                folder_path=str(local_ckpt_dir),
                repo_id=repo_id,
                repo_type="model",
                token=token,
                path_in_repo=path_in_repo,
            )
            # Update marker file with the latest checkpoint name
            try:
                from io import BytesIO

                latest_payload = BytesIO(str(ckpt_name).encode("utf-8"))
                api.upload_file(
                    path_or_fileobj=latest_payload,
                    path_in_repo=f"{path_in_repo_prefix}/latest.txt",
                    repo_id=repo_id,
                    repo_type="model",
                    token=token,
                )
            except Exception:
                ...
        except Exception:
            ...

        # Optional remote pruning: keep recent N checkpoints in the remote repo
        keep = keep_n
        if keep is None:
            try:
                keep = int(os.getenv("EASYDEL_HF_KEEP_N", "0"))
            except Exception:
                keep = 0
        if keep and keep > 0:
            try:
                _prune_remote_old_checkpoints(repo_id=repo_id, token=token, prefix=path_in_repo_prefix, keep_n=int(keep))
            except Exception:
                ...

    # Run upload in background to avoid blocking TPU hosts
    sync = os.getenv("EASYDEL_HF_SYNC") in {"1", "true", "True"}
    if sync:
        _worker()
        return

    t = threading.Thread(target=_worker, daemon=True)
    t.start()
    try:
        timeout = float(os.getenv("EASYDEL_HF_TIMEOUT", "2"))
    except Exception:
        timeout = 2.0
    # Briefly wait so small uploads finish, then continue training regardless
    try:
        t.join(timeout=timeout)
    except Exception:
        ...


def get_latest_checkpoint_dirname(
    *, repo_id: str, token: str, path_in_repo_prefix: str = "checkpoints"
) -> str | None:
    """Return the latest checkpoint folder name (e.g., run-1234) or None."""
    try:
        from huggingface_hub import HfApi, hf_hub_download
    except Exception:
        return None

    api = HfApi()
    try:
        files = api.list_repo_files(repo_id=repo_id, repo_type="model", token=token)
        marker = f"{path_in_repo_prefix}/latest.txt"
        if marker in files:
            try:
                p = hf_hub_download(repo_id=repo_id, filename=marker, repo_type="model", token=token)
                latest_name = EasyPath(p).read_text().strip()
                if latest_name:
                    return latest_name
            except Exception:
                ...

        tree = api.list_repo_tree(
            repo_id=repo_id,
            repo_type="model",
            token=token,
            path=path_in_repo_prefix,
            recursive=False,
        )
        run_names: list[str] = []
        for item in tree or []:
            try:
                rel = item.path.split("/")[-1]
                if rel.startswith("run-"):
                    run_names.append(rel)
            except Exception:
                ...

        def _extract_step(name: str) -> int:
            try:
                return int(re.sub(r"^run-", "", name))
            except Exception:
                return -1

        if run_names:
            run_names.sort(key=_extract_step)
            return run_names[-1]
    except Exception:
        return None
    return None


def download_latest_checkpoint(
    *,
    repo_id: str,
    token: str,
    path_in_repo_prefix: str = "checkpoints",
    local_root: str | os.PathLike | None = None,
) -> EasyPathLike | None:
    """Download the latest checkpoint folder from HF Hub into local_root/run-*/ and return its path."""
    latest_dirname = get_latest_checkpoint_dirname(
        repo_id=repo_id,
        token=token,
        path_in_repo_prefix=path_in_repo_prefix,
    )
    if latest_dirname is None:
        return None
    try:
        from huggingface_hub import HfApi
    except Exception:
        return None
    api = HfApi()
    root = EasyPath(local_root or EasyPath.mktempdir(prefix="easydel-hf-"))
    target = root / latest_dirname
    target.mkdir(parents=True, exist_ok=True)
    try:
        api.download_folder(
            repo_id=repo_id,
            repo_type="model",
            path_in_repo=f"{path_in_repo_prefix}/{latest_dirname}",
            local_dir=str(target),
            token=token,
        )
        return tp.cast(EasyPathLike, target)
    except Exception:
        return None


def _prune_remote_old_checkpoints(*, repo_id: str, token: str, prefix: str, keep_n: int) -> None:
    """Delete remote checkpoints beyond the most recent `keep_n`.

    Operates best-effort; ignores errors.
    """
    try:
        from huggingface_hub import HfApi
    except Exception:
        return

    api = HfApi()
    try:
        tree = api.list_repo_tree(repo_id=repo_id, repo_type="model", token=token, path=prefix, recursive=False)
        run_names: list[str] = []
        for item in tree or []:
            try:
                rel = item.path.split("/")[-1]
                if rel.startswith("run-"):
                    run_names.append(rel)
            except Exception:
                ...
        if len(run_names) <= keep_n:
            return

        def _extract_step(name: str) -> int:
            try:
                return int(re.sub(r"^run-", "", name))
            except Exception:
                return -1

        run_names.sort(key=_extract_step)
        to_delete = run_names[:-keep_n]
        for name in to_delete:
            try:
                api.delete_file(
                    repo_id=repo_id,
                    repo_type="model",
                    path=f"{prefix}/{name}",
                    token=token,
                )
            except Exception:
                ...
    except Exception:
        return


