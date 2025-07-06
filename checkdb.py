import sqlite3

# Connect to your database
conn = sqlite3.connect('instance/users.db')
cursor = conn.cursor()

# List all tables
cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
tables = cursor.fetchall()

print("Tables in the database:", tables)

# Optional: Check columns in 'user' table (if it's there)
if ('user',) in tables:
    cursor.execute("PRAGMA table_info(user);")
    columns = cursor.fetchall()
    print("\nColumns in 'user' table:")
    for col in columns:
        print(col)
else:
    print("\n'user' table not found.")

conn.close()
