from src.database.database import initialise_database


def main() -> None:
    initialise_database()
    print("Database initialized successfully: campusresolve.db")


if __name__ == "__main__":
    main()