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

# 2. 9 reads x 2 refs = 18 rows (+ header).
ROWS=$(tail -n +2 "$OUT/dedup.tsv" | wc -l)
[ "$ROWS" -eq 18 ] && pass "18 per-read rows" || fail "expected 18 rows, got $ROWS"

# 3. Dedup actually happened: 9 reads, 6 unique sequences (r01,r02,r03,r08 share one; r04,r05 share one).
if grep -q "6 unique" "$OUT/dedup.log"; then pass "log reports 6 unique sequences"; else fail "log does not report 6 unique sequences"; grep -i unique "$OUT/dedup.log" || true; fi

# 4. Mixed-case read aligns like its uppercase twin.
C1=$(awk -F'\t' '$1=="r01" && $2=="ampA"{print $3 $4}' "$OUT/dedup.tsv")
C8=$(awk -F'\t' '$1=="r08" && $2=="ampA"{print $3 $4}' "$OUT/dedup.tsv")
[ "$C1" = "$C8" ] && [ "$C1" = "100=+" ] && pass "r01/r08 both 100= on + strand" || fail "r01=$C1 r08=$C8"

# 5. RC read reported on - strand with a perfect CIGAR.
C6=$(awk -F'\t' '$1=="r06" && $2=="ampA"{print $3 $4}' "$OUT/dedup.tsv")
[ "$C6" = "100=-" ] && pass "r06 is 100= on - strand" || fail "r06=$C6"

# 6. Deletion allele.
C4=$(awk -F'\t' '$1=="r04" && $2=="ampA"{print $3}' "$OUT/dedup.tsv")
[ "$C4" = "40=3D57=" ] && pass "r04 is 40=3D57=" || fail "r04=$C4"

# 7. Over-length read is reported as * with strand . and a warning.
C9=$(awk -F'\t' '$1=="r09" && $2=="ampA"{print $3 $4}' "$OUT/dedup.tsv")
[ "$C9" = "*." ] && pass "r09 (130 bp) reported as * ." || fail "r09=$C9"
grep -q "max-read-length" "$OUT/dedup.log" && pass "over-length warning logged" || fail "no over-length warning"

# 8. Collapse mode: header ends in n_reads; counts sum to 9 per reference; (A, CELL1) group has 3 reads (r01, r02, r08).
HDR=$(head -1 "$OUT/collapse.tsv" | awk -F'\t' '{print $NF}')
[ "$HDR" = "n_reads" ] && pass "collapse header ends with n_reads" || fail "collapse header last column is $HDR"
SUM=$(awk -F'\t' 'NR>1 && $2=="ampA"{s+=$NF} END{print s}' "$OUT/collapse.tsv")
[ "$SUM" -eq 9 ] && pass "collapse n_reads sum to 9 for ampA" || fail "collapse n_reads sum for ampA is $SUM"
N=$(awk -F'\t' 'NR>1 && $1=="r01" && $2=="ampA"{print $NF}' "$OUT/collapse.tsv")
[ "$N" = "3" ] && pass "r01 group (ampA WT, CELL1) has n_reads=3" || fail "r01 group n_reads=$N"

echo "=== Results: $PASS passed, $FAIL failed ==="
[ "$FAIL" -eq 0 ]
