from __future__ import annotations

from src.database.database import DATABASE_URL, initialise_database


def main() -> None:
    initialise_database()
    print(f"Database initialized successfully: {DATABASE_URL}")


if __name__ == "__main__":
    main()