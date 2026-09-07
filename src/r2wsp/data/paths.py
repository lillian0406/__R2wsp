from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


def _parse_simple_yaml(path: Path) -> dict[str, str]:
    data: dict[str, str] = {}
    if not path.exists():
        return data

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        key = key.strip()
        value = value.strip().strip("'\"")
        if key:
            data[key] = value
    return data


def find_project_root(start: str | Path | None = None) -> Path:
    here = Path(start) if start is not None else Path(__file__).resolve()
    here = here.resolve()
    if here.is_file():
        here = here.parent

    for candidate in (here, *here.parents):
        if (candidate / "pyproject.toml").exists() and (candidate / "src" / "r2wsp").exists():
            return candidate
    raise FileNotFoundError("could not locate R2wsp project root from current path")


@dataclass(frozen=True)
class DataPaths:
    project_root: Path
    config_path: Path
    data_root: Path
    raw_svs_root: Path
    raw_rna_root: Path
    raw_clinical_root: Path
    token_root: Path

    def as_dict(self) -> dict[str, str]:
        return {
            "project_root": str(self.project_root),
            "config_path": str(self.config_path),
            "data_root": str(self.data_root),
            "raw_svs_root": str(self.raw_svs_root),
            "raw_rna_root": str(self.raw_rna_root),
            "raw_clinical_root": str(self.raw_clinical_root),
            "token_root": str(self.token_root),
        }

    def missing_paths(self) -> dict[str, Path]:
        missing: dict[str, Path] = {}
        for name in ("data_root", "raw_svs_root", "raw_rna_root", "raw_clinical_root", "token_root"):
            path = getattr(self, name)
            if not path.exists():
                missing[name] = path
        return missing

    def validate(self) -> None:
        missing = self.missing_paths()
        if missing:
            details = ", ".join(f"{name}={path}" for name, path in missing.items())
            raise FileNotFoundError(f"missing required data paths: {details}")


def resolve_data_paths(
    *,
    project_root: str | Path | None = None,
    config_path: str | Path | None = None,
    data_root: str | Path | None = None,
) -> DataPaths:
    root = find_project_root(project_root)
    cfg_path = Path(config_path).resolve() if config_path is not None else root / "configs" / "data_paths.yaml"
    cfg = _parse_simple_yaml(cfg_path)

    env_data_root = os.environ.get("R2WSP_DATA_ROOT")
    chosen_data_root = Path(
        data_root
        if data_root is not None
        else env_data_root
        if env_data_root
        else cfg.get("data_root", root / "data")
    )
    data_root_path = chosen_data_root if chosen_data_root.is_absolute() else (root / chosen_data_root)
    data_root_path = data_root_path.resolve()

    def _pick_path(key: str, default: Path) -> Path:
        raw_value = cfg.get(key)
        candidate = Path(raw_value) if raw_value else default
        return candidate if candidate.is_absolute() else (root / candidate)

    raw_svs_root = _pick_path("raw_svs_root", data_root_path / "raw_svs").resolve()
    raw_rna_root = _pick_path("raw_rna_root", data_root_path / "raw_rna").resolve()
    raw_clinical_root = _pick_path("raw_clinical_root", data_root_path / "raw_clinical").resolve()
    token_root = _pick_path("token_root", data_root_path / "tokens").resolve()

    return DataPaths(
        project_root=root,
        config_path=cfg_path,
        data_root=data_root_path,
        raw_svs_root=raw_svs_root,
        raw_rna_root=raw_rna_root,
        raw_clinical_root=raw_clinical_root,
        token_root=token_root,
    )

