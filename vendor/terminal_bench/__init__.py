"""Minimal Terminal-Bench runtime used by the rLLM training environment."""

# Importing the upstream package root eagerly loads optional UI, database, and
# model-provider dependencies. Training imports concrete runtime modules
# directly, so package initialization remains side-effect free.
