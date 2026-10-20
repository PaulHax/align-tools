"""Display metadata from recorded open-world Hydra and meta files."""

from typing import Any, Dict, Optional

from pydantic import BaseModel, ConfigDict


class OpenWorldMetadata(BaseModel):
    """Names for display, independent of the configuration used for identity."""

    model_config = ConfigDict(frozen=True)

    adm_name: str = "open_world"
    llm_name: Optional[str] = None
    driver_class: Optional[str] = None


def resolve_metadata(config: Dict[str, Any], meta: Dict[str, Any]) -> OpenWorldMetadata:
    adm = _as_dict(config.get("adm"))
    driver = _as_dict(config.get("driver"))
    driver_class = _optional_str(driver.get("_target_"))
    architecture = _optional_str(meta.get("architecture")) or _optional_str(
        driver.get("architecture")
    )
    driver_name = (
        _optional_str(meta.get("driver"))
        or (driver_class.rsplit(".", 1)[-1] if driver_class else None)
        or "open_world"
    )
    extra_tools = _as_dict(driver.get("extra_tools"))
    fallback = " + ".join([driver_name, *sorted(extra_tools)])
    return OpenWorldMetadata(
        adm_name=_optional_str(adm.get("name")) or architecture or fallback,
        llm_name=_model_name(adm, driver),
        driver_class=driver_class,
    )


def _model_name(adm: Dict[str, Any], driver: Dict[str, Any]) -> Optional[str]:
    # The driver resolves a non-null model before its chat_model block.
    if driver.get("model") is not None:
        return _optional_str(driver["model"])
    return _optional_str(_as_dict(driver.get("chat_model")).get("model")) or (
        _optional_str(
            _as_dict(adm.get("structured_inference_engine")).get("model_name")
        )
    )


def _as_dict(value: Any) -> Dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _optional_str(value: Any) -> Optional[str]:
    return value if isinstance(value, str) else None
