# SM clock and power over each run's measured window only: from the client timing epoch to the
# last request's return (the .clk samples are 1 s apart, so a 7-14 s window holds 7-14 of them).
# usage: clocks_window.py <json...>   (each with its .clk beside it)
import datetime, json, statistics as st, sys
for f in sys.argv[1:]:
    d = json.load(open(f))
    t0 = d["summary"]["config"]["client timing epoch unix s"]
    t1 = t0 + max(r["client_return_s"] for r in d["requests"])
    s = []
    for line in open(f[:-5] + ".clk"):
        p = [x.strip() for x in line.split(",")]
        try:
            t = datetime.datetime.strptime(p[0], "%Y/%m/%d %H:%M:%S.%f").replace(tzinfo=datetime.timezone.utc).timestamp()
            mhz, w = float(p[1].split()[0]), float(p[2].split()[0])
        except (IndexError, ValueError):
            continue
        if t0 <= t <= t1:
            s.append((mhz, w))
    print(f"{f.split('/')[-1][:-5]:28s} window {t1 - t0:5.1f} s  samples {len(s):3d}  "
          f"SM median {st.median(m for m, _ in s):6.0f} MHz (min {min(m for m, _ in s):.0f})  "
          f"power median {st.median(w for _, w in s):5.1f} W")
