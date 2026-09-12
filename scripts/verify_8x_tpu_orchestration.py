import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).parent.resolve()))
from verify_tpu_orchestration import main

if __name__ == "__main__":
    main()

