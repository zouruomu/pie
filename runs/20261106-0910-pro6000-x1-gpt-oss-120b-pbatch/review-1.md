# Review 1 (orchestrator, 12:55Z)

I recomputed the baseline tables (every cell, medians, ranges, n), the engine sha256s and the pf256 component totals from results/. They all match. One request:

1. **Settle "no regression since ea800efeb" for 20b.** The site's 20b pie number (49.8k at ea800efeb) is 6% above main's best here (47.0k). Today's check is one round at pf8–pf32, where rounds spread ±15%, so it can't rule out a regression, and a ≥4.5% regression would be a W4A16 fix in scope.
   - Run ea800efeb vs main at 20b pf32, pf256 and pf1024: 3 alternating rounds each, same prompts file.
   - Record both engines' sha256 and pie's prompt token count, in case ea800efeb rendered a different prompt length.
   - Report the medians and spreads in report.md, and say plainly whether a regression ≥4.5% exists at any of those points.
   - If one does, stop and tell me in status.md before bisecting. If not, set READY FOR REVIEW again. About 30 minutes.
