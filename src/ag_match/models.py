"""Model spec resolution.

Accepts a pydantic-ai `Model` instance or a string. Strings follow pydantic-ai's
`provider:model` form (`google:gemini-3.8-flash`, `google-cloud:gemini-3.8-flash`,
`openai:gpt-5`, `anthropic:...`).

Google has two routes:

- `google:<name>` is the Gemini Developer API, authenticated with an API key
  (`GOOGLE_API_KEY`).
- `google-cloud:<name>` (alias `vertex:<name>`) is Vertex AI, authenticated with a
  service account or any other Google Cloud credential. Pass a `GoogleCloudAuth`, or
  rely on Application Default Credentials (`GOOGLE_APPLICATION_CREDENTIALS`,
  `gcloud auth application-default login`, or a GCE/GKE metadata server).

Conveniences: `gemini` alone means the default Gemini model, and `gemini:<name>` picks
the route for you: the Developer API when `GOOGLE_API_KEY` (or `GEMINI_API_KEY`) is set
and no `GoogleCloudAuth` is given, otherwise Vertex AI through ADC. ADC finds, in order,
`GOOGLE_APPLICATION_CREDENTIALS`, the gcloud user credential file, and the default
service account of a GCE, GKE, Cloud Run or Cloud Functions host via the metadata server.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from pydantic_ai.models import Model, infer_model

if TYPE_CHECKING:
    from google.auth.credentials import Credentials

DEFAULT_GEMINI_MODEL = "gemini-3.8-flash"
DEFAULT_MODEL = f"google:{DEFAULT_GEMINI_MODEL}"


@dataclass(frozen=True)
class GoogleCloudAuth:
    """How to reach Vertex AI. Give one credential source, or none for ADC.

    Args:
        service_account_file: Path to a service-account JSON key file.
        service_account_info: The parsed contents of such a key file.
        credentials: Any ready `google.auth.credentials.Credentials` object.
        project: Google Cloud project for quota. Defaults to the service account's
            `project_id`, then `GOOGLE_CLOUD_PROJECT`.
        location: Vertex location such as `us-central1` or `global`. Defaults to
            `GOOGLE_CLOUD_LOCATION`, then `us-central1`.
    """

    service_account_file: str | Path | None = None
    service_account_info: dict[str, Any] | None = None
    credentials: Credentials | None = None
    project: str | None = None
    location: str | None = None

    def __post_init__(self) -> None:
        given = [
            x
            for x in (self.service_account_file, self.service_account_info, self.credentials)
            if x is not None
        ]
        if len(given) > 1:
            raise ValueError(
                "Give only one of service_account_file, service_account_info, credentials"
            )

    def resolve_credentials(self) -> tuple[Credentials | None, str | None]:
        """Return (credentials, project). Both may be None, meaning use ADC."""
        from google.oauth2 import service_account

        info = self.service_account_info
        if self.service_account_file is not None:
            info = json.loads(Path(self.service_account_file).read_text(encoding="utf-8"))
        if info is not None:
            creds = service_account.Credentials.from_service_account_info(info)
            return creds, self.project or info.get("project_id")
        return self.credentials, self.project


def _api_key_available() -> bool:
    return bool(os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY"))


_NO_GOOGLE_CREDENTIALS = (
    "No Google credentials found for {spec!r}. Either set GOOGLE_API_KEY for the Gemini "
    "Developer API, or provide Vertex AI credentials: pass google_cloud=GoogleCloudAuth(...), "
    "set GOOGLE_APPLICATION_CREDENTIALS to a service-account key, run "
    "`gcloud auth application-default login`, or run on a Google Cloud host with an attached "
    "service account."
)


def resolve_model(
    spec: str | Model | None, *, google_cloud: GoogleCloudAuth | None = None
) -> Model:
    if isinstance(spec, Model):
        return spec
    if spec is None or spec == "gemini":
        spec = f"gemini:{DEFAULT_GEMINI_MODEL}"

    provider, sep, name = spec.partition(":")
    auto_routed = provider == "gemini"
    if auto_routed:
        provider = "google" if (google_cloud is None and _api_key_available()) else "google-cloud"
    elif provider == "vertex":
        provider = "google-cloud"
    spec = f"{provider}{sep}{name}"

    if provider == "google-cloud" and google_cloud is not None:
        from pydantic_ai.models.google import GoogleModel
        from pydantic_ai.providers.google_cloud import GoogleCloudProvider

        credentials, project = google_cloud.resolve_credentials()
        return GoogleModel(
            name,
            provider=GoogleCloudProvider(
                credentials=credentials,
                project=project or os.getenv("GOOGLE_CLOUD_PROJECT"),
                location=google_cloud.location or os.getenv("GOOGLE_CLOUD_LOCATION"),
            ),
        )
    if provider == "google-cloud":
        from google.auth.exceptions import DefaultCredentialsError

        try:
            return infer_model(spec)
        except DefaultCredentialsError as exc:
            raise DefaultCredentialsError(_NO_GOOGLE_CREDENTIALS.format(spec=spec)) from exc
    return infer_model(spec)
