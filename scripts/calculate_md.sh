#!/usr/bin/env bash
set -euo pipefail

GMX=${GMX:-gmx_mpi}
TEMP=${TEMP:-298.15}
FF=${FF:-amber14sb}
WATER=${WATER:-tip3p}
BOX_DIST=${BOX_DIST:-1.2}
THREADS=${THREADS:-16}
NSTEPS=${NSTEPS:-50000000}   # 100 ns at 2 fs

INPUT_PDB=${1:-complex_clean.pdb}

echo "[1] Generate topology"
$GMX pdb2gmx \
  -f "$INPUT_PDB" \
  -o complex_processed.gro \
  -p topol.top \
  -i posre.itp \
  -ff "$FF" \
  -water "$WATER" \
  -ignh

echo "[2] Define box"
$GMX editconf \
  -f complex_processed.gro \
  -o complex_newbox.gro \
  -c \
  -d "$BOX_DIST" \
  -bt dodecahedron

echo "[3] Solvate"
$GMX solvate \
  -cp complex_newbox.gro \
  -cs spc216.gro \
  -o complex_solv.gro \
  -p topol.top

cat > ions.mdp << 'EOF'
integrator      = steep
emtol           = 1000.0
emstep          = 0.01
nsteps          = 50000
cutoff-scheme   = Verlet
nstlist         = 20
rlist           = 1.0
coulombtype     = PME
rcoulomb        = 1.0
vdwtype         = Cut-off
rvdw            = 1.0
pbc             = xyz
EOF

echo "[4] Add ions"
$GMX grompp \
  -f ions.mdp \
  -c complex_solv.gro \
  -p topol.top \
  -o ions.tpr

echo "SOL" | $GMX genion \
  -s ions.tpr \
  -o complex_solv_ions.gro \
  -p topol.top \
  -pname NA \
  -nname CL \
  -neutral \
  -conc 0.15

cat > minim.mdp << 'EOF'
integrator      = steep
emtol           = 1000.0
emstep          = 0.01
nsteps          = 50000
cutoff-scheme   = Verlet
nstlist         = 20
rlist           = 1.0
coulombtype     = PME
rcoulomb        = 1.0
vdwtype         = Cut-off
rvdw            = 1.0
pbc             = xyz
EOF

echo "[5] Energy minimization"
$GMX grompp \
  -f minim.mdp \
  -c complex_solv_ions.gro \
  -p topol.top \
  -o em.tpr

$GMX mdrun -deffnm em -ntomp "$THREADS" -pin on

cat > nvt.mdp << EOF
define              = -DPOSRES
integrator          = md
dt                  = 0.002
nsteps              = 250000
nstxout-compressed  = 5000
nstenergy           = 1000
nstlog              = 1000
continuation        = no
constraint_algorithm = lincs
constraints         = h-bonds
lincs_iter          = 1
lincs_order         = 4
cutoff-scheme       = Verlet
nstlist             = 20
rlist               = 1.0
coulombtype         = PME
rcoulomb            = 1.0
vdwtype             = Cut-off
rvdw                = 1.0
DispCorr            = EnerPres
tcoupl              = V-rescale
tc-grps             = System
tau_t               = 0.1
ref_t               = $TEMP
pcoupl              = no
pbc                 = xyz
gen_vel             = yes
gen_temp            = $TEMP
gen_seed            = -1
EOF

echo "[6] NVT"
$GMX grompp \
  -f nvt.mdp \
  -c em.gro \
  -r em.gro \
  -p topol.top \
  -o nvt.tpr

$GMX mdrun -deffnm nvt -ntomp "$THREADS" -pin on

cat > npt.mdp << EOF
define              = -DPOSRES
integrator          = md
dt                  = 0.002
nsteps              = 500000
nstxout-compressed  = 5000
nstenergy           = 1000
nstlog              = 1000
continuation        = yes
constraint_algorithm = lincs
constraints         = h-bonds
lincs_iter          = 1
lincs_order         = 4
cutoff-scheme       = Verlet
nstlist             = 20
rlist               = 1.0
coulombtype         = PME
rcoulomb            = 1.0
vdwtype             = Cut-off
rvdw                = 1.0
DispCorr            = EnerPres
tcoupl              = V-rescale
tc-grps             = System
tau_t               = 0.1
ref_t               = $TEMP
pcoupl              = C-rescale
pcoupltype          = isotropic
tau_p               = 5.0
ref_p               = 1.0
compressibility     = 4.5e-5
pbc                 = xyz
gen_vel             = no
EOF

echo "[7] NPT"
$GMX grompp \
  -f npt.mdp \
  -c nvt.gro \
  -r nvt.gro \
  -t nvt.cpt \
  -p topol.top \
  -o npt.tpr

$GMX mdrun -deffnm npt -ntomp "$THREADS" -pin on

cat > md.mdp << EOF
integrator          = md
dt                  = 0.002
nsteps              = $NSTEPS
nstxout-compressed  = 5000
nstenergy           = 5000
nstlog              = 5000
continuation        = yes
constraint_algorithm = lincs
constraints         = h-bonds
lincs_iter          = 1
lincs_order         = 4
cutoff-scheme       = Verlet
nstlist             = 20
rlist               = 1.0
coulombtype         = PME
rcoulomb            = 1.0
vdwtype             = Cut-off
rvdw                = 1.0
DispCorr            = EnerPres
tcoupl              = V-rescale
tc-grps             = System
tau_t               = 0.1
ref_t               = $TEMP
pcoupl              = Parrinello-Rahman
pcoupltype          = isotropic
tau_p               = 5.0
ref_p               = 1.0
compressibility     = 4.5e-5
pbc                 = xyz
gen_vel             = no
EOF

echo "[8] Production MD"
$GMX grompp \
  -f md.mdp \
  -c npt.gro \
  -t npt.cpt \
  -p topol.top \
  -o md.tpr

$GMX mdrun \
  -deffnm md \
  -ntomp "$THREADS" \
  -pin on \
  -nb gpu \
  -pme gpu