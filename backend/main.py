"""Entrypoint for Vercel Serverless Functions and local ASGI servers."""

from app.main import app

__all__ = ["app"]
