import sqlite3, sys
db, start_idx, n = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
c = sqlite3.connect(db)
rows = c.execute("""select k.start, k.end, s.value, k.gridX, k.gridY, k.gridZ, k.blockX, k.blockY, k.blockZ, k.streamId, k.graphNodeId, k.graphId, d.value
 from CUPTI_ACTIVITY_KIND_KERNEL k join StringIds s on s.id=k.shortName join StringIds d on d.id=k.demangledName order by k.start""").fetchall()
t0 = rows[start_idx][0]
prev_end = None
for i in range(start_idx, min(len(rows), start_idx + n)):
    st, en, nm, gx, gy, gz, bx, by, bz, sid, gn, gid, dm = rows[i]
    gap = (st - prev_end) / 1000 if prev_end else 0
    print(f"{i:6d} t={(st-t0)/1000:9.2f} dur={(en-st)/1000:7.2f} gap={gap:6.2f} s{sid} g{gid} [{gx},{gy},{gz}]x[{bx},{by},{bz}] {nm} | {dm[:150]}")
    prev_end = en if prev_end is None else max(prev_end, en)
