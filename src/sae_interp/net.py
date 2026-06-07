"""Network / TLS setup helpers.

Why this exists
---------------
The libraries here (transformers, datasets, sae-lens) download models, SAEs and
corpora from the Hugging Face Hub over HTTPS. On a corporate laptop the traffic
often goes through a TLS-inspecting proxy (e.g. Netskope, Zscaler) that presents
its *own* root CA. Python's bundled ``certifi`` CA store doesn't know that CA, so
downloads fail with::

    [SSL: CERTIFICATE_VERIFY_FAILED] self-signed certificate in certificate chain

The fix is to make Python trust the *operating system* certificate store (the
macOS keychain / Windows cert store), where the corporate CA is already installed.
The ``truststore`` package does exactly that. We call it once, early, and it makes
``ssl`` (and therefore httpx/requests/urllib) use the OS trust store.

This is a no-op if ``truststore`` isn't installed, so the module is safe to import
everywhere. It does NOT bypass certificate verification — it just uses a different,
OS-managed source of trusted roots.

Note: this only fixes *TLS trust*. If your network *blocks* Hugging Face entirely
(a web-filter category block, returning HTTP 403 with an ``X-Direct-Response``
header), no client-side change helps — you need a network exception or a different
network. See README "Network / corporate proxy" for that case.
"""

from __future__ import annotations

import os

_INJECTED = False


def enable_os_trust_store() -> bool:
    """Make Python's SSL use the OS trust store. Returns True if it took effect.

    Idempotent and safe to call multiple times. Set the env var
    ``SAE_INTERP_NO_TRUSTSTORE=1`` to skip (e.g. if it ever conflicts with your env).
    """
    global _INJECTED
    if _INJECTED:
        return True
    if os.environ.get("SAE_INTERP_NO_TRUSTSTORE") == "1":
        return False
    try:
        import truststore  # type: ignore

        truststore.inject_into_ssl()
        _INJECTED = True
        return True
    except Exception:
        # truststore not installed or failed — fall back to default certifi behaviour.
        return False


def load_dotenv_if_present() -> None:
    """Load environment variables from a local ``.env`` (HF_TOKEN, WANDB_API_KEY).

    No-op if python-dotenv isn't installed or there's no .env file.
    """
    try:
        from dotenv import load_dotenv  # type: ignore

        load_dotenv()
    except Exception:
        pass


def bootstrap() -> None:
    """One-call setup every entry-point script runs first: load .env + fix TLS trust."""
    load_dotenv_if_present()
    enable_os_trust_store()
