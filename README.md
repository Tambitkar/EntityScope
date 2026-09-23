# EntityScope
# Data

This folder contains the custom entity patterns used by the EntityScope Named Entity Recognition (NER) application.

## Files

### `ansp-clean-patterns.jsonl`

This file contains the custom Entity Ruler patterns used by the spaCy NLP pipeline. These patterns help the application identify domain-specific entities such as:

* **TAXA** — Taxonomic names of biological organisms or species.
* **HABITAT** — Places or environments where organisms live.

The application loads these patterns into spaCy's `EntityRuler` during text processing.

## Usage

The file is automatically loaded by the main application. No manual processing of this file is required when running the application.
