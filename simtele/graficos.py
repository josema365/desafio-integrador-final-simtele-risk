"""Gráficos SVG sin dependencias, generados desde los datos recuperados de la base de datos."""
import math

PALETA = ["#1f3b73", "#d62728", "#2ca02c", "#ff7f0e", "#9467bd", "#17becf", "#8c564b", "#7f7f7f"]


def _ticks_lin(lo, hi, n=5):
    if hi == lo:
        hi = lo + 1
    return [lo + (hi - lo) * i / n for i in range(n + 1)]


def _fmt(v):
    if v == 0:
        return "0"
    return f"{v:.3g}" if 1e-3 <= abs(v) < 1e5 else f"{v:.1e}"


def svg_lineas(series, xlabel="", ylabel="", log_y=False, log_x=False, ancho=640, alto=300):
    """series: {nombre: dict(x=[...], y=[...], estilo='linea'|'puntos')}."""
    ml, mr, mt, mb = 58, 12, 26, 40
    datos = {}
    for n, s in series.items():
        pts = [(x, y) for x, y in zip(s.get("x") or [], s.get("y") or [])
               if x is not None and y is not None and math.isfinite(x) and math.isfinite(y)]
        if log_x:
            pts = [(x, y) for x, y in pts if x > 0]
        if pts:
            datos[n] = (pts, s.get("estilo", "linea"))
    if not datos:
        return "<p class='nota'>Sin datos suficientes para graficar.</p>"
    fy = (lambda v: math.log10(max(abs(v), 1e-17))) if log_y else (lambda v: v)
    fx = (lambda v: math.log10(v)) if log_x else (lambda v: v)
    xs = [fx(x) for pts, _ in datos.values() for x, _ in pts]; ys = [fy(y) for pts, _ in datos.values() for _, y in pts]
    x0, x1, y0, y1 = min(xs), max(xs), min(ys), max(ys)
    if log_y:
        y0, y1 = math.floor(y0), math.ceil(y1)
        if y1 == y0: y1 += 1
    elif y1 == y0:
        y0, y1 = y0 - 1, y1 + 1
    else:
        pad = 0.05 * (y1 - y0); y0, y1 = y0 - pad, y1 + pad
    if x1 == x0: x1 = x0 + 1
    X = lambda v: ml + (ancho - ml - mr) * (fx(v) - x0) / (x1 - x0)
    Y = lambda v: mt + (alto - mt - mb) * (y1 - fy(v)) / (y1 - y0)
    o = [f'<svg viewBox="0 0 {ancho} {alto}" xmlns="http://www.w3.org/2000/svg" class="grafico" role="img" aria-label="{ylabel} vs {xlabel}">']
    # rejilla y ejes
    yt = [10.0 ** d for d in range(int(y0), int(y1) + 1)] if log_y else _ticks_lin(y0, y1)
    for v in yt:
        yy = Y(v)
        o.append(f'<line x1="{ml}" x2="{ancho-mr}" y1="{yy:.1f}" y2="{yy:.1f}" stroke="#e3e7ef"/><text x="{ml-6}" y="{yy+3:.1f}" font-size="9" text-anchor="end">{_fmt(v)}</text>')
    if log_x:
        vals = sorted({x for pts, _ in datos.values() for x, _ in pts})
        xt = vals if len(vals) <= 7 else [vals[int(i * (len(vals) - 1) / 6)] for i in range(7)]
    else:
        xt = _ticks_lin(x0, x1)
    for v in xt:
        xx = X(v)
        o.append(f'<line x1="{xx:.1f}" x2="{xx:.1f}" y1="{mt}" y2="{alto-mb}" stroke="#f0f2f7"/><text x="{xx:.1f}" y="{alto-mb+13}" font-size="9" text-anchor="middle">{_fmt(v)}</text>')
    o.append(f'<line x1="{ml}" x2="{ml}" y1="{mt}" y2="{alto-mb}" stroke="#555"/><line x1="{ml}" x2="{ancho-mr}" y1="{alto-mb}" y2="{alto-mb}" stroke="#555"/>'
             f'<text x="{(ml+ancho-mr)/2:.0f}" y="{alto-6}" font-size="10" text-anchor="middle">{xlabel}</text>'
             f'<text x="12" y="{(mt+alto-mb)/2:.0f}" font-size="10" text-anchor="middle" transform="rotate(-90 12 {(mt+alto-mb)/2:.0f})">{ylabel}</text>')
    lx, ly = ml + 6, 10
    for i, (n, (pts, est)) in enumerate(datos.items()):
        c = PALETA[i % len(PALETA)]
        if est == "puntos":
            o += [f'<circle cx="{X(x):.1f}" cy="{Y(y):.1f}" r="3.2" fill="{c}" fill-opacity=".85"/>' for x, y in pts]
        else:
            o.append(f'<polyline fill="none" stroke="{c}" stroke-width="1.6" points="' + " ".join(f"{X(x):.1f},{Y(y):.1f}" for x, y in pts) + '"/>')
            if len(pts) <= 12:
                o += [f'<circle cx="{X(x):.1f}" cy="{Y(y):.1f}" r="2.2" fill="{c}"/>' for x, y in pts]
        if lx + 8 * len(n) + 22 > ancho - mr:
            lx, ly = ml + 6, ly + 11
        o.append(f'<rect x="{lx}" y="{ly-6}" width="9" height="3" fill="{c}"/><text x="{lx+12}" y="{ly}" font-size="9">{n}</text>')
        lx += 22 + 5.2 * len(n)
    o.append("</svg>")
    return "".join(o)


def graficos_de_simulacion(ejs):
    """Construye (curvas, [grupos de error]) a partir de las ejecuciones cargadas de la base.
    Devuelve una lista de dicts {titulo, svg}."""
    out = []
    ref = next((e["referencia"] for e in ejs if e.get("referencia", {}).get("curva")), {})
    series = {}
    if ref.get("curva"):
        series[ref.get("curva_nombre", "referencia")] = dict(ref["curva"])
    for n, s in (ref.get("curvas_extra") or {}).items():
        series[n] = dict(s)
    for e in ejs:
        c = (e.get("valor") or {}).get("curva")
        if c:
            extra = ""
            nc = (e["valor"] or {}).get("nivel_curva")
            if nc:
                extra = f" (n={nc['n']})"
            series[e["metodo"] + extra] = dict(c)
    for n, s in (ref.get("puntos") or {}).items():
        series[n] = dict(s, estilo="puntos")
    eje = ref.get("eje_curvas", {})
    if series:
        out.append(dict(titulo="Solución / escenario", svg=svg_lineas(series, eje.get("xlabel", "x"), eje.get("ylabel", "y"))))
    grupos = {}
    for e in ejs:
        se = (e.get("valor") or {}).get("serie_error")
        if se and se.get("x"):
            grupos.setdefault(se["xlabel"], {})[e["metodo"]] = dict(se)
    for xl, ser in grupos.items():
        log_x = any(s.get("log_x") for s in ser.values())
        out.append(dict(titulo=f"Error / convergencia — {xl}", svg=svg_lineas(ser, xl, "error absoluto", log_y=True, log_x=log_x)))
    return out
