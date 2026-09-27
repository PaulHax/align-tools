"""Preserve the source run's launch configuration in MLflow artifact storage."""

import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Dict

from align_utils.open_world import CONFIG_FILE, META_FILE, OpenWorldRun
from mlflow import MlflowClient

from .traces import RUN_KEY_TAG, run_key, run_label

PROVENANCE_FILES = (
    CONFIG_FILE,
    Path(".hydra/overrides.yaml"),
    Path(".hydra/hydra.yaml"),
    Path(META_FILE),
)
_DIGESTS_TAG = "align.provenance_sha256"


def read_provenance(run_dir: Path) -> Dict[str, bytes]:
    """Snapshot available files before writing anything to MLflow."""
    return {
        path.as_posix(): (run_dir / path).read_bytes()
        for path in PROVENANCE_FILES
        if (run_dir / path).is_file()
    }


def preserve_run(experiment_id: str, run: OpenWorldRun, files: Dict[str, bytes]) -> str:
    """Reuse a source run and copy each provenance file once, without overwrites.

    One importer must own a source run at a time. A digest is recorded only after
    the artifact write succeeds, so interrupted uploads can safely be retried.
    """
    client = MlflowClient()
    key = run_key(run)
    matches = client.search_runs(
        experiment_ids=[experiment_id],
        filter_string=f"tags.`{RUN_KEY_TAG}` = '{key}'",
        max_results=2,
    )
    if len(matches) > 1:
        raise ValueError(f"Multiple MLflow runs have source key {key}")
    source = (
        matches[0]
        if matches
        else client.create_run(
            experiment_id,
            run_name=run_label(run),
            tags={
                RUN_KEY_TAG: key,
                "align.source_path": str(run.path.resolve()),
                **{
                    name: value
                    for name, value in {
                        "align.source_version": run.version,
                        "adm": run.config.adm.name if run.config else None,
                        "llm": run.config.adm.llm_backbone if run.config else None,
                        "adm_profile": run.adm_profile,
                    }.items()
                    if value
                },
            },
        )
    )
    saved = json.loads(source.data.tags.get(_DIGESTS_TAG, "{}"))
    digests = {
        name: hashlib.sha256(content).hexdigest() for name, content in files.items()
    }
    for name, digest in digests.items():
        if name in saved and saved[name] != digest:
            raise ValueError(
                f"Source provenance changed: {name}; "
                "the archived file was kept. Import into a new experiment "
                "if this is a different run."
            )
    with TemporaryDirectory() as staging:
        for name, content in files.items():
            if name in saved:
                continue
            path = Path(staging) / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(content)
            client.log_artifact(
                source.info.run_id,
                str(path),
                artifact_path=(Path("source") / name).parent.as_posix(),
            )
            saved[name] = digests[name]
            client.set_tag(source.info.run_id, _DIGESTS_TAG, json.dumps(saved))
    return source.info.run_id
