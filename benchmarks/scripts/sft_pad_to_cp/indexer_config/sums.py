import json

rows = json.load(open("grid.json"))
st = {(r["layout"], r["total"], r["rank"]): r for r in rows if r["k_pad_to"] == 1}
pd = {(r["layout"], r["total"], r["rank"]): r for r in rows if r["k_pad_to"] == 16}
best_st = sum(min(r["ms"].values()) for r in st.values())
print(f"stock per-shape best sum {best_st:.1f} ms")
for c in st[next(iter(st))]["ms"]:
    s = sum(r["ms"][c] for r in st.values())
    p = sum(r["ms"][c] for r in pd.values())
    wa_s = max(r["ms"][c] - min(r["ms"].values()) for r in st.values())
    wa_p = max(pd[k]["ms"][c] - min(pd[k]["ms"].values()) for k in pd)
    worst_vs_stock = max(pd[k]["ms"][c] / min(st[k]["ms"].values()) for k in pd)
    print(
        f"{c:16s} stock sum {s:7.1f} (worst +{wa_s:5.2f} ms)  padded sum {p:7.1f} (worst +{wa_p:5.2f} ms vs padded best,"
        f" worst ratio vs stock best {worst_vs_stock:.3f})"
    )
