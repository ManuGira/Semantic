# Running Python Scripts with UV

This project uses **`uv`** to manage Python scripts instead of the traditional `python` command.

## Usage

To run any Python script in this project, use:

```bash
uv run myscript.py
```

**NOT:**
```bash
python myscript.py
```

## Examples

- `uv run src/main_semantic_game.py`
- `uv run src/main_assistant.py`
- `uv run src/main_cluster.py`

For any automation or AI agent execution, always use the `uv run` command format.
