import sqlite3
import commons

def getprefixed1(prefix, num, item): return "{}{}_{}".format(prefix, num, item)

def getprefixed0(prefix, num, items): return ", ".join(
    ["{}{}.{} AS {}".format(prefix, num, i, getprefixed1(prefix, num, i)) for i in items])

def getprefixed(prefixprefix, upto, changing, indicators, predicted, skip_predicted=False):
    all = [getprefixed0(prefixprefix, i, changing + indicators) for i in range(1, upto)]
    if not skip_predicted: all.append(getprefixed0(prefixprefix, upto, changing + predicted))
    return ", ".join(all)

def getjoinrange(prefixprefix, upto): return " JOIN bank_data ".join(
    [getjoinitem(prefixprefix, i) for i in range(2, upto + 1)])

def getjoinitem(prefixprefix, num): return prefixprefix + str(
    num) + " ON " + prefixprefix + "1" + ".ncodpers=" + prefixprefix + str(
    num) + ".ncodpers AND date(" + prefixprefix + str(
    num) + ".fecha_dato) = date(" + prefixprefix + "1" + ".fecha_dato, '+" + str(num - 1) + " months')"

def notnull(prefix, num, items):
    return " AND ".join("{} IS NOT NULL ".format(getprefixed1(prefix, num, i)) for i in items)

change_in_time = ['age', 'antiguedad', 'indrel', 'indrel_1mes', 'tiprel_1mes', 'indresi', 'indext', 'indfall',
                  'tipodom', 'cod_prov', 'ind_actividad_cliente', 'segmento']

constant_in_time = ['ind_empleado', 'pais_residencia', 'sexo', 'ind_nuevo', 'canal_entrada', 'renta']

# predicted = ['ind_cco_fin_ult1', 'ind_cno_fin_ult1', 'ind_tjcr_fin_ult1', 'ind_nomina_ult1', 'ind_nom_pens_ult1',
#             'ind_recibo_ult1']

def getlatestproductsquery(ncodpers):
    return "SELECT {} FROM bank_data where ncodpers={} and fecha_dato='2016-05-28'".format(", ".join(commons.indicators), ncodpers)

def getquery(prefix, span, limit):
    return "SELECT " + getprefixed0(prefix, 1, constant_in_time) + ", /**/ " + getprefixed(prefix, span,
                                                                                           change_in_time,
                                                                                           commons.indicators,
                                                                                           commons.indicators) + " FROM bank_data bd1 JOIN bank_data " + getjoinrange(
        "bd",
        span) + " WHERE bd1.fecha_dato >= date('2015-01-28') and bd1.fecha_dato <= date('2016-05-28', '-" + str(
        span - 1) + " months') AND bd1_ind_empleado IS NOT NULL AND " + notnull(prefix, span,
                                                                                commons.indicators) + " ORDER BY RANDOM() LIMIT " + str(
        limit) + ";"

def getquery_for_ncodpers(prefix, span, ncodpers):
    return "SELECT " + getprefixed0(prefix, 1, ['ncodpers'] + constant_in_time) + ", /**/ " + getprefixed(prefix, span,
                                                                                                          change_in_time,
                                                                                                          commons.indicators,
                                                                                                          commons.indicators,
                                                                                                          False) + " FROM bank_data bd1 JOIN bank_data " + getjoinrange(
        "bd", span) + " WHERE bd1.fecha_dato = date('2015-12-28') AND bd1.ncodpers in " + "({})".format(
        ", ".join([str(n) for n in ncodpers]))

if __name__ == "__main__":
    print(".mode csv")
    print(".headers on")
    print(".output " + commons.main_csv)
    print("pragma cache_size 2000000;")
    print(getquery("bd", commons.num_months, commons.chunk_size))
    print(".quit")
