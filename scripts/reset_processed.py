#!/usr/bin/env python3
"""
Script to reset the 'processed' column to 0 for all rows in the videos table.
"""

import sqlite3
import os

# Path to the database file
DB_PATH = os.path.join(os.path.dirname(__file__), 'videos_cache.db')

def reset_processed_column():
    """Reset all processed values to 0 in the videos table."""
    try:
        # Connect to the database
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()

        # Update all rows to set processed = 0
        cursor.execute("UPDATE videos SET processed = 0")

        # Get the number of rows affected
        rows_affected = cursor.rowcount

        # Commit the changes
        conn.commit()

        print(f"Successfully reset {rows_affected} rows to processed = 0")

        # Verify the update
        cursor.execute("SELECT COUNT(*) FROM videos WHERE processed = 0")
        count = cursor.fetchone()[0]
        print(f"Verification: {count} rows now have processed = 0")

    except sqlite3.Error as e:
        print(f"Database error: {e}")
        return 1
    except Exception as e:
        print(f"Error: {e}")
        return 1
    finally:
        if conn:
            conn.close()

    return 0

if __name__ == "__main__":
    exit(reset_processed_column())
