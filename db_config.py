import os
import psycopg2
from dotenv import load_dotenv

load_dotenv()


def get_db_params() -> dict:
    """
    Returns database connection parameters from environment variables.
    Raises ValueError if database password is not configured.
    """
    password = (
        os.getenv("DB_PASS")
        or os.getenv("DB_PASSWORD")
        or os.getenv("POSTGRES_PASSWORD")
    )
    if not password:
        raise ValueError(
            "Database password not configured. Please set DB_PASS, DB_PASSWORD, or POSTGRES_PASSWORD in .env or environment."
        )

    return {
        "host": os.getenv("DB_HOST", "localhost"),
        "port": int(os.getenv("DB_PORT", "5432")),
        "database": os.getenv("DB_NAME", "bitcoin_analyst"),
        "user": os.getenv("DB_USER", "postgres"),
        "password": password,
    }


def get_db_connection():
    """Returns an open psycopg2 database connection."""
    return psycopg2.connect(**get_db_params())
