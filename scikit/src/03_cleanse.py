import sqlite3, sys

def fix_by_finding_earlier_or_later_int(conn, col):
    print("Processing {}...".format(col))
    cur = conn.cursor()

    wontfix = 0
    fixed = 0

    for row in cur.execute("SELECT fecha_dato, ncodpers FROM bank_data WHERE typeof({}) <> 'integer'".format(col)):

        fecha_dato = row[0]
        ncodpers = row[1]

        c2 = conn.cursor()
        c2.execute(
            "SELECT {} FROM bank_data WHERE typeof({}) = 'integer' AND ncodpers = ? AND fecha_dato < ? ORDER BY fecha_dato DESC LIMIT 1".format(
                col, col),
            (ncodpers, fecha_dato))
        newval = c2.fetchone()

        if newval is None:
            c2.execute(
                "SELECT {} FROM bank_data WHERE typeof({}) = 'integer' AND ncodpers = ? AND fecha_dato > ? ORDER BY fecha_dato DESC LIMIT 1".format(
                    col, col),
                (ncodpers, fecha_dato))
            newval = c2.fetchone()

        if newval is not None:
            fixed = fixed + 1
            c2.execute("UPDATE bank_data SET {}=? WHERE fecha_dato=? and ncodpers=?".format(col),
                       (newval[0], fecha_dato, ncodpers))
        else:
            wontfix = wontfix + 1

    print("Done with {}. Wontfix={}, fixed={}".format(col, wontfix, fixed))

def drop_clients_with_only_one_entry(conn):
    cur = conn.cursor()
    print("Dropping clients with just a sinlge month of data...")
    cur.execute("DELETE FROM bank_data WHERE ncodpers IN (SELECT ncodpers FROM bank_data GROUP BY ncodpers HAVING count(1) = 1);")
    print("Done. Deleted {} rows.".format(cur.rowcount))

def fix_age(conn):
    cur = conn.cursor()
    print("Erase ages < 20 and > 99")
    cur.execute("UPDATE bank_data SET age = NULL WHERE age < 20 OR age > 99")
    print("Updated: {}".format(cur.rowcount))
    print("Get average age")
    cur.execute("SELECT AVG(age) FROM bank_data WHERE age IS NOT NULL")
    avgage=int(cur.fetchone()[0])
    print("Set missing age to mean={}".format(avgage))
    cur.execute("UPDATE bank_data SET age = ? WHERE age IS NULL", (avgage,))

with sqlite3.connect('./santander_v2.db', detect_types=sqlite3.PARSE_DECLTYPES) as conn:

    conn.execute("PRAGMA cache_size = -8000000")

    fix_age(conn)
    drop_clients_with_only_one_entry(conn)

    fix_by_finding_earlier_or_later_int(conn, 'age')
    fix_by_finding_earlier_or_later_int(conn, 'ind_nomina_ult1')
    fix_by_finding_earlier_or_later_int(conn, 'ind_nom_pens_ult1')

    conn.commit()
