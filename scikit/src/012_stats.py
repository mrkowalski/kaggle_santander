import sqlite3

with sqlite3.connect('./santander_v2.db', detect_types=sqlite3.PARSE_DECLTYPES) as conn:
    conn.execute("PRAGMA cache_size = -8000000")
    cur = conn.cursor()
    cur.execute('SELECT age, indfall, count(1) FROM bank_data GROUP BY age, indfall')
    print(str(cur.fetchall()))
