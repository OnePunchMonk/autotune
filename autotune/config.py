"""Secrets/config loading. Defaults to .env; can be swapped to AWS AppConfig
by setting AUTOTUNE_SECRETS_PROVIDER=appconfig so the harness can be deployed
without a .env file baked into the image.
"""
from __future__ import annotations

import json
import os
from functools import lru_cache

from dotenv import load_dotenv

load_dotenv()


@lru_cache(maxsize=1)
def _appconfig_snapshot() -> dict:
    import boto3

    client = boto3.client("appconfigdata", region_name=os.environ.get("AWS_REGION", "us-east-1"))
    app = os.environ["AWS_APPCONFIG_APPLICATION"]
    env = os.environ["AWS_APPCONFIG_ENVIRONMENT"]
    profile = os.environ["AWS_APPCONFIG_PROFILE"]

    session = client.start_configuration_session(
        ApplicationIdentifier=app,
        EnvironmentIdentifier=env,
        ConfigurationProfileIdentifier=profile,
    )
    resp = client.get_latest_configuration(ConfigurationToken=session["InitialConfigurationToken"])
    return json.loads(resp["Configuration"].read())


def get_secret(name: str, default: str | None = None, required: bool = False) -> str | None:
    provider = os.environ.get("AUTOTUNE_SECRETS_PROVIDER", "dotenv")

    value = None
    if provider == "appconfig":
        value = _appconfig_snapshot().get(name) or os.environ.get(name)
    else:
        value = os.environ.get(name)

    value = value if value is not None else default
    if required and not value:
        raise RuntimeError(
            f"Missing required secret '{name}'. Set it in .env (see .env.example) "
            f"or in AWS AppConfig if AUTOTUNE_SECRETS_PROVIDER=appconfig."
        )
    return value


def anthropic_api_key() -> str:
    return get_secret("ANTHROPIC_API_KEY", required=True)


def agent_model() -> str:
    return get_secret("AUTOTUNE_AGENT_MODEL", default="claude-sonnet-5")
