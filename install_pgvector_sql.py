"""Load an extracted pgvector extension script into the configured database (one-off fallback).

Prefer ``CREATE EXTENSION vector`` after installing pgvector into PostgreSQL. This script is only for a
server where the extension files cannot be installed: it runs the extension SQL with MODULE_PATHNAME
replaced by the given library path.

    python install_pgvector_sql.py --sql-file <dir>/share/extension/vector--0.8.6.sql \
        --module-path <PostgreSQL>/lib/vector
"""
import argparse

import db_config


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--sql-file", required=True, help="extracted vector--<version>.sql")
    ap.add_argument("--module-path", required=True, help="pgvector library path without extension, e.g. <PostgreSQL>/lib/vector")
    args = ap.parse_args()

    with open(args.sql_file, "r", encoding="utf-8") as f:
        lines = f.readlines()

    clean_lines = [line for line in lines if not line.strip().startswith("\\")]
    clean_sql = "".join(clean_lines)

    # Replace MODULE_PATHNAME with explicit path
    clean_sql = clean_sql.replace("MODULE_PATHNAME", args.module_path)

    conn = db_config.get_db_connection()
    conn.autocommit = True
    cur = conn.cursor()

    try:
        cur.execute(clean_sql)
        print("[SUCCESS] pgvector types, operators, and HNSW index access method created successfully in PostgreSQL!")
    except Exception as e:
        print("[ERROR] Failed to execute:", e)
    finally:
        cur.close()
        conn.close()


if __name__ == "__main__":
    main()
