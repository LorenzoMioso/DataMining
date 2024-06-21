# Required python version: 3.11

# Create virtual environment

```bash
python -m venv .env
```

# Activate virtual environment

```bash
source .env/bin/activate
```

# Install dependencies

```bash
pip install -r requirements.txt
```

# Fix pygraphviz installation for macM1

```bash
python3 -m pip install -U --no-cache-dir  \
        --config-settings="--global-option=build_ext" \
        --config-settings="--global-option=-I$(brew --prefix graphviz)/include/" \
        --config-settings="--global-option=-L$(brew --prefix graphviz)/lib/" \
        pygraphviz
```