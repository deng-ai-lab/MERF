#!/usr/bin/env bash
set -euo pipefail

GMX=${GMX:-gmx_mpi}

# ===== 用户必须修改这三项 =====
# 例：抗体 H+L vs antigen
# REC_RI=("1-220" "221-435")
# LIG_RI=("436-680")

# 例：TCRαβ vs pMHC
# REC_RI=("1-210" "211-430")
# LIG_RI=("431-706" "707-716" "717-815")

REC_RI=(${REC_RI:-"1-243"})
LIG_RI=(${LIG_RI:-"244-250"})

FIRST_NEW_GROUP=${FIRST_NEW_GROUP:-17}

START_PS=${START_PS:-50000}      # 正式建议跳过前 20-50 ns；单位 ps
END_PS=${END_PS:-100000}         # 单位 ps
DT_PS=${DT_PS:-100}              # MM/GBSA 抽帧间隔；正式可 50-200 ps

REC_GROUP_NAME=${REC_GROUP_NAME:-Receptor}
LIG_GROUP_NAME=${LIG_GROUP_NAME:-Ligand}
COM_GROUP_NAME=${COM_GROUP_NAME:-Complex}

echo "[1] Generate index file"

idx=$FIRST_NEW_GROUP
rec_ids=()
lig_ids=()

rm -f make_index.in

for r in "${REC_RI[@]}"; do
  echo "ri $r" >> make_index.in
  echo "name $idx REC_${idx}" >> make_index.in
  rec_ids+=("$idx")
  idx=$((idx+1))
done

if [ "${#rec_ids[@]}" -eq 1 ]; then
  rec_group="${rec_ids[0]}"
else
  rec_group="${rec_ids[0]}"
  for ((i=1; i<${#rec_ids[@]}; i++)); do
    echo "$rec_group | ${rec_ids[$i]}" >> make_index.in
    echo "name $idx REC_tmp_${idx}" >> make_index.in
    rec_group="$idx"
    idx=$((idx+1))
  done
fi
echo "name $rec_group $REC_GROUP_NAME" >> make_index.in

for r in "${LIG_RI[@]}"; do
  echo "ri $r" >> make_index.in
  echo "name $idx LIG_${idx}" >> make_index.in
  lig_ids+=("$idx")
  idx=$((idx+1))
done

if [ "${#lig_ids[@]}" -eq 1 ]; then
  lig_group="${lig_ids[0]}"
else
  lig_group="${lig_ids[0]}"
  for ((i=1; i<${#lig_ids[@]}; i++)); do
    echo "$lig_group | ${lig_ids[$i]}" >> make_index.in
    echo "name $idx LIG_tmp_${idx}" >> make_index.in
    lig_group="$idx"
    idx=$((idx+1))
  done
fi
echo "name $lig_group $LIG_GROUP_NAME" >> make_index.in

echo "$rec_group | $lig_group" >> make_index.in
com_group="$idx"
echo "name $com_group $COM_GROUP_NAME" >> make_index.in
echo "q" >> make_index.in

$GMX make_ndx \
  -f md.tpr \
  -o index.ndx < make_index.in

echo "Receptor group ID = $rec_group"
echo "Ligand group ID   = $lig_group"
echo "Complex group ID  = $com_group"

echo "[2] Remove PBC and output Complex only"

echo "$com_group" | $GMX trjconv \
  -s md.tpr \
  -f md.xtc \
  -o md_complex_noPBC.xtc \
  -pbc mol \
  -center \
  -ur compact \
  -n index.ndx

echo "[3] Fit trajectory to receptor and output Complex only"

printf "%s\n%s\n" "$rec_group" "$com_group" | $GMX trjconv \
  -s md.tpr \
  -f md_complex_noPBC.xtc \
  -o md_complex_fit.xtc \
  -fit rot+trans \
  -n index.ndx

echo "[4] Extract equilibrated window"

echo "$com_group" | $GMX trjconv \
  -s md.tpr \
  -f md_complex_fit.xtc \
  -o md_complex_fit_window.xtc \
  -b "$START_PS" \
  -e "$END_PS" \
  -n index.ndx

echo "[5] Downsample frames"

echo "$com_group" | $GMX trjconv \
  -s md.tpr \
  -f md_complex_fit_window.xtc \
  -o md_complex_fit_window_dt.xtc \
  -dt "$DT_PS" \
  -n index.ndx

echo "[6] Generate reference PDB from one complex frame"

echo "$com_group" | $GMX trjconv \
  -s md.tpr \
  -f md_complex_fit_window_dt.xtc \
  -o md_ref_complex.pdb \
  -dump "$START_PS" \
  -n index.ndx

echo "[7] Write MM/GBSA input"

cat > mmpbsa_gbneck2.in << 'EOF'
&general
  sys_name="complex_MMGBSA",
  startframe=1,
  endframe=999999,
  interval=1,
  verbose=1,
  keep_files=0,
  PBRadii=4,
/
&gb
  igb=8,
  saltcon=0.150,
  intdiel=6.0,
  extdiel=80.0,
/
EOF

echo "[8] Run gmx_MMPBSA"

rm -rf _GMXMMPBSA*

gmx_MMPBSA \
  -O \
  -nogui \
  -i mmpbsa_gbneck2.in \
  -cs md.tpr \
  -ct md_complex_fit_window_dt.xtc \
  -ci index.ndx \
  -cg "$rec_group" "$lig_group" \
  -cp topol.top \
  -o FINAL_RESULTS_MMPBSA.dat \
  -eo FINAL_RESULTS_MMPBSA.csv

echo "[9] Extract key result"

grep -A 30 -n "Delta" FINAL_RESULTS_MMPBSA.dat || true
grep "_TOTAL" FINAL_RESULTS_MMPBSA.dat || true