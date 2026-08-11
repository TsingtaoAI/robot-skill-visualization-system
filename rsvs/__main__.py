"""python -m newtest → 打开 hub 说明页。"""

import newtest.bootstrap  # noqa: F401

from newtest.hub import main

if __name__ == "__main__":
    main()
