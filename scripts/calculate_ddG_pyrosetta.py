#!/usr/bin/env python3
"""
Calculate ddG (binding free energy change) for antibody-antigen complexes using PyRosetta.
"""

import os
import sys
import argparse
import json
from pathlib import Path
import time

import pyrosetta
from pyrosetta.rosetta.protocols.analysis import InterfaceAnalyzerMover
from pyrosetta.rosetta.protocols.relax import FastRelax
from pyrosetta.rosetta.core.pack.task import TaskFactory
from pyrosetta.rosetta.core.pack.task import operation
from pyrosetta.rosetta.core.select import residue_selector as selections
from pyrosetta.rosetta.core.select.movemap import MoveMapFactory, move_map_action

import numpy as np
from Bio import PDB


def init_pyrosetta():
    """Initialize PyRosetta with appropriate options.

    Origin: Merged from RADAb baselines:
    - energy.py:7-15 (basic init flags)
    - pyrosetta_relaxer.py:10-20 (additional flags like -constant_seed)
    """
    pyrosetta.init(' '.join([
        '-mute', 'all',
        '-use_input_sc',
        '-ignore_unrecognized_res',
        '-ignore_zero_occupancy', 'false',
        '-load_PDB_components', 'false',
        '-relax:default_repeats', '2',
        '-no_fconfig',
        '-constant_seed',
    ]))
    print("[INFO] PyRosetta initialized successfully")


def relax_structure(pdb_path, output_path, flexible_region_first=None, flexible_region_last=None, max_iter=1000):
    """
    Relax protein structure using FastRelax.

    Args:
        pdb_path: Path to input PDB file
        output_path: Path to save relaxed PDB
        flexible_region_first: Tuple (chain, resseq) for start of flexible region (default: relax all)
        flexible_region_last: Tuple (chain, resseq) for end of flexible region
        max_iter: Maximum iterations for FastRelax

    Returns:
        energy_before, energy_after
    """
    
    # Load structure
    pose = pyrosetta.pose_from_pdb(pdb_path)
    original_pose = pose.clone()

    # Create score function
    scorefxn = pyrosetta.create_score_function('ref2015')

    # Setup FastRelax
    relax = FastRelax()
    relax.set_scorefxn(scorefxn)
    relax.max_iter(max_iter)

    # Setup task factory - only allow repacking, no design
    tf = TaskFactory()
    tf.push_back(operation.InitializeFromCommandline())
    tf.push_back(operation.RestrictToRepacking())

    # If flexible region specified, restrict relaxation to that region
    if flexible_region_first is not None and flexible_region_last is not None:
        try:
            chain_first, resseq_first = flexible_region_first
            chain_last, resseq_last = flexible_region_last

            # Create selector for flexible region
            flex_selector = selections.ResidueIndexSelector()
            pose_first = pose.pdb_info().pdb2pose(chain_first, resseq_first)
            pose_last = pose.pdb_info().pdb2pose(chain_last, resseq_last)
            flex_selector.set_index_range(pose_first, pose_last)

            # Neighborhood selector to include nearby residues
            nbr_selector = selections.NeighborhoodResidueSelector()
            nbr_selector.set_focus_selector(flex_selector)
            nbr_selector.set_include_focus_in_subset(True)

            # Prevent repacking outside neighborhood
            prevent_repacking_rlt = operation.PreventRepackingRLT()
            prevent_outside = operation.OperateOnResidueSubset(
                prevent_repacking_rlt,
                nbr_selector,
                flip_subset=True,
            )
            tf.push_back(prevent_outside)

            # Setup movemap
            mmf = MoveMapFactory()
            mmf.add_bb_action(move_map_action.mm_enable, flex_selector)
            mmf.add_chi_action(move_map_action.mm_enable, nbr_selector)
            mm = mmf.create_movemap_from_pose(pose)
            relax.set_movemap(mm)

            print(f"[RELAX] Flexible region: {chain_first}{resseq_first} - {chain_last}{resseq_last}")
        except Exception as e:
            print(f"[WARN] Could not set flexible region: {e}. Relaxing entire structure.")

    relax.set_task_factory(tf)

    # Relax
    relax.apply(pose)

    # Calculate energies
    energy_before = scorefxn(original_pose)
    energy_after = scorefxn(pose)

    print(f"[RELAX] Energy before: {energy_before:.4f}")
    print(f"[RELAX] Energy after:  {energy_after:.4f}")

    # Save relaxed structure
    pose.dump_pdb(output_path)
    print(f"[RELAX] Saved relaxed structure: {output_path}")

    return energy_before, energy_after


def calculate_interface_energy(pdb_path, interface_str):
    """
    Calculate interface binding energy (dG) using InterfaceAnalyzerMover.

    Args:
        pdb_path: Path to PDB file
        interface_str: Interface definition, e.g., "HL_A" (antibody chains HL vs antigen chain A)

    Returns:
        dG_separated: Interface binding free energy (kcal/mol)
    """
    print(f"[ENERGY] Calculating interface energy for: {os.path.basename(pdb_path)}")
    print(f"[ENERGY] Interface: {interface_str}")

    pose = pyrosetta.pose_from_pdb(pdb_path)

    # Create interface analyzer
    mover = InterfaceAnalyzerMover(interface_str)
    mover.set_pack_separated(True)

    # Apply analyzer
    mover.apply(pose)

    # Extract binding energy
    dG = pose.scores['dG_separated']

    print(f"[ENERGY] dG_separated: {dG:.4f} kcal/mol")

    return dG


def calculate_ddG(wt_pdb, mut_pdb, interface_str, output_dir, relax_wt=True, relax_mut=True, verbose=False):
    """
    Calculate ddG = dG_mut - dG_wt.

    Args:
        wt_pdb: Path to wildtype PDB
        mut_pdb: Path to mutant PDB
        interface_str: Interface definition (e.g., "HL_A")
        output_dir: Directory to save results
        relax_wt: Whether to relax wildtype structure
        relax_mut: Whether to relax mutant structure
        verbose: Whether to print detailed progress information

    Returns:
        results dict with ddG and individual energies
    """

    # Initialize PyRosetta, if not already initialized
    init_pyrosetta()

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    results = {
        'wt_pdb': os.path.abspath(wt_pdb),
        'mut_pdb': os.path.abspath(mut_pdb),
        'interface': interface_str,
        'output_dir': output_dir,
    }

    # Relax structures if requested
    wt_pdb_relax = os.path.join(output_dir, 'wt_relaxed.pdb')
    mut_pdb_relax = os.path.join(output_dir, 'mut_relaxed.pdb')

    if relax_wt:
        print("\n" + "="*70)
        print("STEP 1: Relaxing wildtype structure")
        print("="*70)
        try:
            start_time = time.time()
            relax_structure(wt_pdb, wt_pdb_relax)
            end_time = time.time()
            print(f"[INFO] WT relaxation time: {end_time - start_time:.2f} seconds")
            results['wt_relaxed'] = True
        except Exception as e:
            print(f"[ERROR] Failed to relax WT: {e}")
            wt_pdb_relax = wt_pdb
            results['wt_relaxed'] = False
    else:
        wt_pdb_relax = wt_pdb
        results['wt_relaxed'] = False

    if relax_mut:
        print("\n" + "="*70)
        print("STEP 2: Relaxing mutant structure")
        print("="*70)
        try:
            start_time = time.time()
            relax_structure(mut_pdb, mut_pdb_relax)
            end_time = time.time()
            print(f"[INFO] Mutant relaxation time: {end_time - start_time:.2f} seconds")
            results['mut_relaxed'] = True
        except Exception as e:
            print(f"[ERROR] Failed to relax mutant: {e}")
            mut_pdb_relax = mut_pdb
            results['mut_relaxed'] = False
    else:
        mut_pdb_relax = mut_pdb
        results['mut_relaxed'] = False

    # Calculate interface energies and ddG
    print("\n" + "="*70)
    print("STEP 3: Calculating interface energies")
    print("="*70)

    start_time = time.time()
    dG_wt = calculate_interface_energy(wt_pdb_relax, interface_str)
    end_time = time.time()
    print(f"[INFO] WT interface energy calculation time: {end_time - start_time:.2f} seconds")
    results['dG_wt'] = float(dG_wt)

    start_time = time.time()
    dG_mut = calculate_interface_energy(mut_pdb_relax, interface_str)
    end_time = time.time()
    print(f"[INFO] Mutant interface energy calculation time: {end_time - start_time:.2f} seconds")
    results['dG_mut'] = float(dG_mut)

    ddG = results['dG_mut'] - results['dG_wt']
    results['ddG'] = float(ddG)

    return results

def main():
    wt_path = "/home/dataset-local/projects_dir/MERF/data/7FAE/PDBs_fixed/7FAE.pdb"
    mut_path = "/home/dataset-local/projects_dir/MERF/data/7FAE/PDBs_mutated/7FAE_AH53G.pdb"
    partner = "HL_A"
    output_dir = "/home/dataset-local/projects_dir/MERF/scripts/docking_test/"

    results = calculate_ddG(wt_path, mut_path, partner, output_dir)
    print(results)

if __name__ == '__main__':
    main()
