"""Prepare an isolated client without changing platform packages."""
from pathlib import Path
import argparse
import subprocess
import sys
import venv


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    assert sys.platform == 'linux'
    assert not args.output.exists(), 'Use a new environment; existing environments are retained'
    venv.EnvBuilder(system_site_packages=True, with_pip=True).create(args.output)
    requirements = Path(__file__).resolve().parents[1]/'requirements-bench-client.txt'
    subprocess.run([str(args.output/'bin/python'), '-m', 'pip', 'install', '--no-deps',
                    '-r', str(requirements)], check=True, timeout=900)
    print(args.output/'bin/python')


if __name__ == '__main__':
    main()
