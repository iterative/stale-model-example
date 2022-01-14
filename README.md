This is the project that accompanies the [Preventing Stale Models in Production blog post]().

## Get started

Here's what you need to do to get this project running.

- Clone the repo
- Create a virtual environment with a command like: `python -m venv .venv`
- Install all of the dependencies with : `pip install -r requirements.txt`
- Download the data for the project from [this data registry](https://github.com/iterative/dataset-registry/tree/master/blog) with: `dvc get https://github.com/iterative/dataset-registry blog/cats-dogs`

Now you should be able to run the project and DVC is already in place.

## Running experiments

All that's left is running experiments. To do that, open a terminal and run: `dvc exp run`