from __future__ import annotations

from cvrag.core.cv_data import rebuild_cv_json


def main() -> None:
    data = rebuild_cv_json()
    if not data:
        raise SystemExit("Failed to extract CV JSON from PDF.")
    print("cv.json generated.")


if __name__ == "__main__":
    main()
