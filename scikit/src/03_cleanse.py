import sqlite3, sys

#Maunal actions########################

#DROP COLUMN tipodom
#DROP COLUMN cod_prov. Keep nomprov as this gives a chance to add more data like city size, etc.
#DROP COLUMN conyuemp - Only one customer has conyuemp = S

#######################################
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

with sqlite3.connect('../santander_v2.db', detect_types=sqlite3.PARSE_DECLTYPES) as conn:

    fix_by_finding_earlier_or_later_int(conn, 'age')
    fix_by_finding_earlier_or_later_int(conn, 'ind_nomina_ult1')
    fix_by_finding_earlier_or_later_int(conn, 'ind_nom_pens_ult1')

    drop_clients_with_only_one_entry(conn)

    conn.commit()
