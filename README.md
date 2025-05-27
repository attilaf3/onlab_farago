# ONLAB – Energiaközösségek modellezése

Ez a projekt egyetemi szakdolgozati munka keretében készült, és célja különböző energiaközösségi rendszerek determinisztikus optimalizálása. A modellek a megújuló energiatermelés, energiatárolás (BESS és hőtároló), valamint vezérelhető hőtermelők (bojler, hőszivattyú) működését vizsgálják órás felbontásban egy teljes évi szimulációval.

A számítások célja a rendszer **környezeti teljesítményének javítása**: a helyben felhasznált megújuló energia növelése, a hálózattól való függés csökkentése, valamint az önfogyasztás és önellátás optimalizálása.

## Tartalom

| Fájl | Leírás |
|------|--------|
| `optimize.py` | Egyfelhasználós modell (PV + ELH + BESS + HSS) optimalizálása |
| `optimize_two_users.py` | Kétfelhasználós közösségi modell optimalizálása (megosztott BESS, külön bojlerek) |
| `optimizecaller.py` | A futások vezérlése, szcenáriók generálása |
| `input.csv` | Idősoros bemeneti adatok (PV, fogyasztás, HMV, időjárás) |
| `plots/` | A vizualizációk: energiamérlegek, SCI/SSI mutatók, éves és szezonális diagramok |
| `docs/` | A szakdolgozat LaTeX-formátumban és PDF-ben (`onlab.pdf`) |

## Röviden a modellekről

- **1 háztartás (PV + ELH)** – alapmodell, kontrollált bojler, hőtárolóval
- **2 háztartás (PV + ELH)** – közös BESS, de külön bojlerek; energia megosztás vizsgálata
- **1 háztartás (PV + ELH + HP)** – hőszivattyús modell, épülethőmérsékletet is optimalizálva (5R2C modell)

A megoldó Gurobi Optimizer, lineáris vagy vegyes-egész MILP modellezéssel.

---

Ha publikusan osztod meg a repót, érdemes még megadni:
- licence típusa (`LICENSE`)
- DOI vagy link az egyetemi dolgozathoz (ha lesz ilyen)

Szólj, ha ezt Markdown-fájlként is kéred `.md` formátumban!
