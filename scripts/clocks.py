# Median SM clock and power while the GPU was busy (samples drawing > 40 W) for each run's .clk file.
# usage: clocks.py <glob...>
import glob, statistics as st, sys
for f in sorted(sum((glob.glob(a) for a in sys.argv[1:]), [])):
    s = []
    for line in open(f):
        p = [x.strip() for x in line.split(",")]
        try:
            mhz, w = float(p[1].split()[0]), float(p[2].split()[0])
        except (IndexError, ValueError):
            continue
        if w > 40:
            s.append((mhz, w))
    if s:
        print(f"{f.split('/')[-1][:-4]:36s} busy {len(s):4d} s  SM median {st.median(m for m, _ in s):6.0f} MHz "
              f"(p10 {sorted(m for m, _ in s)[len(s)//10]:.0f})  power median {st.median(w for _, w in s):5.1f} W")
