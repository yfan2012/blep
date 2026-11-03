#!/bin/bash
# End-to-end smoke test for blep deduplication and collapse output.
# Requires a CUDA GPU and a release build (cargo build --release).
set -euo pipefail
cd "$(dirname "$0")"

BLEP=${BLEP:-./target/release/blep}
READS=test_data/dedup_reads.fastq
REFS=test_data/refs.fa
OUT=${OUT:-test_data/out}
mkdir -p "$OUT"

FLAGS=(--max-read-length 120 --max-ref-length 120 --max-score 100 -g 5 -e 1 -m 3 --header-tag "RG:Z:")

# Expected CIGARs are derived from the ampA length so the data file stays the single source of truth.
LEN=$(awk 'NR==2{print length($0)}' "$REFS")
DEL_TAIL=$((LEN - 43))   # r04/r05 delete bases 41-43 of ampA

PASS=0; FAIL=0
pass() { echo "  PASS: $1"; PASS=$((PASS + 1)); }
fail() { echo "  FAIL: $1"; FAIL=$((FAIL + 1)); }

echo "=== blep dedup tests ==="
"$BLEP" -r "$READS" -R "$REFS" -o "$OUT/dedup.tsv"    "${FLAGS[@]}" -v 2> "$OUT/dedup.log"
"$BLEP" -r "$READS" -R "$REFS" -o "$OUT/nodedup.tsv"  "${FLAGS[@]}" --no-dedup -v 2> "$OUT/nodedup.log"
"$BLEP" -r "$READS" -R "$REFS" -o "$OUT/collapse.tsv" "${FLAGS[@]}" --collapse -v 2> "$OUT/collapse.log"

# 1. Dedup is lossless: identical rows (order may differ between runs).
if diff <(sort "$OUT/dedup.tsv") <(sort "$OUT/nodedup.tsv") > /dev/null; then
    pass "dedup output identical to --no-dedup output"
else
    fail "dedup output differs from --no-dedup output"; diff <(sort "$OUT/dedup.tsv") <(sort "$OUT/nodedup.tsv") | head
fi

# 2. 11 records (9 reads + one unmerged pair) x 2 refs = 22 rows (+ header).
ROWS=$(tail -n +2 "$OUT/dedup.tsv" | wc -l)
[ "$ROWS" -eq 22 ] && pass "22 per-read rows" || fail "expected 22 rows, got $ROWS"

# 3. Dedup actually happened: 11 records, 7 unique sequences
#    (r01,r02,r03,r08 = ampA WT incl. the lowercase copy; r04,r05 = deletion allele; r06 = RC;
#     r07 = ampB; r09 = long read; r10 = two distinct mate sequences).
if grep -q "7 unique" "$OUT/dedup.log"; then pass "log reports 7 unique sequences"; else fail "log does not report 7 unique sequences"; grep -i unique "$OUT/dedup.log" || true; fi

# 4. Mixed-case read aligns like its uppercase twin.
C1=$(awk -F'\t' '$1=="r01" && $2=="ampA"{print $3 $4}' "$OUT/dedup.tsv")
C8=$(awk -F'\t' '$1=="r08" && $2=="ampA"{print $3 $4}' "$OUT/dedup.tsv")
[ "$C1" = "$C8" ] && [ "$C1" = "${LEN}=+" ] && pass "r01/r08 both ${LEN}= on + strand" || fail "r01=$C1 r08=$C8 (expected ${LEN}=+)"

# 5. RC read reported on - strand with a perfect CIGAR.
C6=$(awk -F'\t' '$1=="r06" && $2=="ampA"{print $3 $4}' "$OUT/dedup.tsv")
[ "$C6" = "${LEN}=-" ] && pass "r06 is ${LEN}= on - strand" || fail "r06=$C6 (expected ${LEN}=-)"

# 6. Deletion allele.
C4=$(awk -F'\t' '$1=="r04" && $2=="ampA"{print $3}' "$OUT/dedup.tsv")
[ "$C4" = "40=3D${DEL_TAIL}=" ] && pass "r04 is 40=3D${DEL_TAIL}=" || fail "r04=$C4 (expected 40=3D${DEL_TAIL}=)"

# 7. Over-length read is reported as * with strand . and a warning.
C9=$(awk -F'\t' '$1=="r09" && $2=="ampA"{print $3 $4}' "$OUT/dedup.tsv")
[ "$C9" = "*." ] && pass "r09 (130 bp) reported as * ." || fail "r09=$C9"
grep -q "max-read-length" "$OUT/dedup.log" && pass "over-length warning logged" || fail "no over-length warning"

# 8. Unmerged pair r10: two per-read rows per reference under one name, one per strand.
STR=$(awk -F'\t' '$1=="r10" && $2=="ampA"{print $4}' "$OUT/dedup.tsv" | sort | tr -d '\n')
[ "$STR" = "+-" ] && pass "r10 mates reported on + and - under one name" || fail "r10 strands: $STR"

# 9. Collapse mode: header ends in n_reads; n_reads summed over rows equals the 11 records per reference;
#    (A, CELL1) group has 3 reads (r01, r02, r08); r10's mates stay together under one name.
HDR=$(head -1 "$OUT/collapse.tsv" | awk -F'\t' '{print $NF}')
[ "$HDR" = "n_reads" ] && pass "collapse header ends with n_reads" || fail "collapse header last column is $HDR"
SUM=$(awk -F'\t' 'NR>1 && $2=="ampA"{s+=$NF} END{print s}' "$OUT/collapse.tsv")
[ "$SUM" -eq 11 ] && pass "collapse n_reads sum to 11 for ampA" || fail "collapse n_reads sum for ampA is $SUM"
N=$(awk -F'\t' 'NR>1 && $1=="r01" && $2=="ampA"{print $NF}' "$OUT/collapse.tsv")
[ "$N" = "3" ] && pass "r01 group (ampA WT, CELL1) has n_reads=3" || fail "r01 group n_reads=$N"
CSTR=$(awk -F'\t' 'NR>1 && $1=="r10" && $2=="ampA"{print $4 $NF}' "$OUT/collapse.tsv" | sort | tr -d '\n')
[ "$CSTR" = "+1-1" ] && pass "collapsed r10 keeps both mates under one name (n_reads=1 each)" || fail "collapsed r10 rows: $CSTR"

# 10. Per-cell counts must agree between per-read and collapsed output (needs the missionbio script).
COUNT=$HOME/Code/missionbio/scripts/count_cigars_per_cell.py
if [ -f "$COUNT" ]; then
    python3 "$COUNT" -i "$OUT/dedup.tsv"    -o "$OUT/dedup.cells.txt"    2> /dev/null
    python3 "$COUNT" -i "$OUT/collapse.tsv" -o "$OUT/collapse.cells.txt" 2> /dev/null
    if cmp -s "$OUT/dedup.cells.txt" "$OUT/collapse.cells.txt"; then
        pass "per-cell counts identical for per-read and collapsed output"
    else
        fail "per-cell counts differ between per-read and collapsed output"; diff "$OUT/dedup.cells.txt" "$OUT/collapse.cells.txt" | head
    fi
else
    echo "  SKIP: $COUNT not found; per-cell count comparison skipped"
fi

echo "=== Results: $PASS passed, $FAIL failed ==="
[ "$FAIL" -eq 0 ]
