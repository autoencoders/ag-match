from __future__ import annotations

import json

import pytest
from google.auth.credentials import AnonymousCredentials
from pydantic_ai.models.google import GoogleModel

from namematch import DEFAULT_MODEL, GoogleCloudAuth, Matcher, resolve_model


def vertex_client(model: GoogleModel):
    return model.client._api_client


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    for var in (
        "GOOGLE_API_KEY",
        "GEMINI_API_KEY",
        "GOOGLE_APPLICATION_CREDENTIALS",
        "GOOGLE_CLOUD_PROJECT",
        "GOOGLE_CLOUD_LOCATION",
    ):
        monkeypatch.delenv(var, raising=False)


@pytest.fixture
def fake_sa(monkeypatch):
    """Route service-account loading to anonymous credentials; key parsing needs a real key."""
    from google.oauth2 import service_account

    seen: list[dict] = []

    def fake_from_info(info, **_):
        seen.append(info)
        return AnonymousCredentials()

    monkeypatch.setattr(service_account.Credentials, "from_service_account_info", fake_from_info)
    return seen


def test_default_is_api_key_route(monkeypatch):
    monkeypatch.setenv("GOOGLE_API_KEY", "k")
    model = resolve_model(None)
    assert model.system == "google"
    assert f"google:{model.model_name}" == DEFAULT_MODEL


def test_gemini_alias_uses_vertex_when_only_adc_env_is_set(monkeypatch, tmp_path, fake_sa):
    key = tmp_path / "sa.json"
    key.write_text(json.dumps({"type": "service_account", "project_id": "adc-proj"}))
    monkeypatch.setenv("GOOGLE_APPLICATION_CREDENTIALS", str(key))
    monkeypatch.setenv("GOOGLE_CLOUD_PROJECT", "adc-proj")
    model = resolve_model("gemini:gemini-3.8-flash")
    assert model.system == "google-cloud"
    assert vertex_client(model).vertexai


def test_gemini_alias_stays_on_api_key_when_key_present(monkeypatch, tmp_path):
    monkeypatch.setenv("GOOGLE_API_KEY", "k")
    monkeypatch.setenv("GOOGLE_APPLICATION_CREDENTIALS", str(tmp_path / "sa.json"))
    assert resolve_model("gemini:gemini-3.8-flash").system == "google"


def test_service_account_file_sets_credentials_and_project(tmp_path, fake_sa):
    key = tmp_path / "sa.json"
    key.write_text(json.dumps({"type": "service_account", "project_id": "file-proj"}))
    model = resolve_model(
        "vertex:gemini-3.8-flash",
        google_cloud=GoogleCloudAuth(service_account_file=key, location="global"),
    )
    assert model.system == "google-cloud"
    client = vertex_client(model)
    assert client.vertexai
    assert client.project == "file-proj"
    assert client.location == "global"
    assert fake_sa[0]["project_id"] == "file-proj"


def test_service_account_info_and_explicit_project_override(fake_sa):
    auth = GoogleCloudAuth(
        service_account_info={"type": "service_account", "project_id": "info-proj"},
        project="override",
    )
    model = resolve_model("gemini:gemini-3.8-flash", google_cloud=auth)
    assert model.system == "google-cloud"
    assert vertex_client(model).project == "override"
    assert vertex_client(model).location == "us-central1"


def test_explicit_credentials_with_env_location(monkeypatch):
    monkeypatch.setenv("GOOGLE_CLOUD_LOCATION", "europe-west1")
    auth = GoogleCloudAuth(credentials=AnonymousCredentials(), project="p")
    model = resolve_model("google-cloud:gemini-3.8-flash", google_cloud=auth)
    assert vertex_client(model).location == "europe-west1"


@pytest.fixture
def fake_metadata_server(monkeypatch):
    """Pretend google.auth.default() found a host-attached default service account."""
    import google.auth

    calls: list[dict] = []

    def fake_default(**kwargs):
        calls.append(kwargs)
        return AnonymousCredentials(), "meta-proj"

    monkeypatch.setattr(google.auth, "default", fake_default)
    return calls


def test_default_service_account_with_no_configuration(fake_metadata_server):
    model = resolve_model("gemini")
    assert model.system == "google-cloud"
    assert model.model_name == "gemini-3.8-flash"
    client = vertex_client(model)
    assert client.vertexai
    assert client.project == "meta-proj"
    assert client.location == "us-central1"
    assert len(fake_metadata_server) == 1


def test_explicit_google_cloud_spec_uses_default_service_account(fake_metadata_server):
    model = resolve_model("google-cloud:gemini-3.8-flash")
    assert vertex_client(model).project == "meta-proj"


def test_no_credentials_anywhere_gives_actionable_error(monkeypatch):
    import google.auth
    from google.auth.exceptions import DefaultCredentialsError

    def fake_default(**_):
        raise DefaultCredentialsError("not found")

    monkeypatch.setattr(google.auth, "default", fake_default)
    with pytest.raises(DefaultCredentialsError, match="GOOGLE_API_KEY.*GoogleCloudAuth"):
        resolve_model("gemini")


def test_only_one_credential_source():
    with pytest.raises(ValueError):
        GoogleCloudAuth(service_account_file="a.json", credentials=AnonymousCredentials())


def test_matcher_accepts_google_cloud(tool):
    auth = GoogleCloudAuth(credentials=AnonymousCredentials(), project="p", location="global")
    matcher = Matcher(tool, model="gemini", google_cloud=auth)
    assert matcher.model.system == "google-cloud"
    assert matcher.model.model_name == "gemini-3.8-flash"


def test_non_google_specs_pass_through(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "k")
    assert resolve_model("openai:gpt-5").system == "openai"
