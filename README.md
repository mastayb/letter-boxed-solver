# letter-boxed-solver

Python solver for NYT Letter Boxed game

## Installation

Follow these steps to install and run the script:

1. Clone the repository:

```bash
git clone https://github.com/yourusername/letter-boxed-solver.git
cd letter-boxed-solver# letter-boxed-solver
```

2. Create a virtual environment and activate it:

```bash
python3 -m venv env
source env/bin/activate  # On Windows, use `env\Scripts\activate`
```

3. Install the required packages

```bash
pip install -r requirements.txt
```

## Usage

To run the script, you need to provide four sets of letters and optionally a path to a dictionary file. If you don't provide a dictionary file, the script will use the default dictionary at `/usr/share/dict/words`.

Here's an example of how to run the script:

```bash
python letter_boxer.py ryd ivf wpn oge --dictionary ../dict/2of12.txt
```

In this example, `ryd`, `ivf`, `wpn`, and `oge` are the four sets of letters. The `--dictionary` option is used to specify a custom dictionary file.

The script will output the optimal solutions to the Letter Boxed puzzle.
