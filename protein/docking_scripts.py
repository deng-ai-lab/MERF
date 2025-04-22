import os


def clear_dir_single(workdir):
    print(f'-- clear the workdir of {workdir} ! --')
    
    output_path = os.path.join(workdir, 'output_files')
    output_relax_input_path = os.path.join(output_path, 'output_relax_input')
    output_docking_path = os.path.join(output_path, 'output_docking')
    output_refine_path = os.path.join(output_path, 'output_refine')
    output_relax_output_path = os.path.join(output_path, 'output_relax_output')
    if os.path.exists(output_path):
        cmd = 'rm -r ' + output_path
        os.system(cmd)
    if not os.path.exists(output_path):
        cmd = 'mkdir ' + output_path
        os.system(cmd)
        
        cmd = 'mkdir ' + output_relax_input_path
        os.system(cmd)
        cmd = 'mkdir ' + output_docking_path
        os.system(cmd)
        cmd = 'mkdir ' + output_refine_path
        os.system(cmd)
        cmd = 'mkdir ' + output_relax_output_path
        os.system(cmd)
    
    print('-- finish clearing ! --')


def docking_simple_single_abbind(workdir, pdb_id, chain, is_parallel, cpu_num=50, is_accurate=False, simple_docking_samples=50,
                          score_key='total_score'):
    print(f'-- start docking pdb:{pdb_id} ! --')
    
    output_path = os.path.join(workdir, 'output_files')
    output_relax_input_path = os.path.join(output_path, 'output_relax_input')
    output_docking_path = os.path.join(output_path, 'output_docking')
    output_refine_path = os.path.join(output_path, 'output_refine')
    output_relax_output_path = os.path.join(output_path, 'output_relax_output')
    
    # -------------------------------- 1. input relax -------------------------------- #
    if is_accurate:
        print(f'-- 1. input relax --')
        
        input_file = os.path.join(workdir, pdb_id + '.pdb')
        
        pdb_relaxed_name = pdb_id + '_0001.pdb'
        pdb_relaxed_temp = os.path.join(output_relax_input_path, pdb_relaxed_name)
        pdb_relaxed = os.path.join(workdir, pdb_id + '_relaxed.pdb')
        score_relax = os.path.join(output_relax_input_path, 'score.sc')
        
        if not os.path.exists(pdb_relaxed_temp):
            if is_parallel:
                cmd = 'mpirun -np ' + str(2) + \
                      ' relax.mpi.linuxgccrelease -in:file:s ' + input_file + \
                      ' -out:path:all ' + output_relax_input_path + \
                      ' -scorefile_format json' + \
                      ' -nstruct 2 -relax:constrain_relax_to_start_coords -relax:ramp_constraints false -ex1 -ex2 -use_input_sc -flip_HNQ -no_optH false -mute all'
            else:
                cmd = 'relax.default.linuxgccrelease -in:file:s ' + input_file + \
                      ' -out:path:all ' + output_relax_input_path + \
                      ' -scorefile_format json' + \
                      ' -nstruct 2 -relax:constrain_relax_to_start_coords -relax:ramp_constraints false -ex1 -ex2 -use_input_sc -flip_HNQ -no_optH false -mute all'
            os.system(cmd)
            
            # choose pdb with lower energy and change the path of pdb_relaxed_temp
            with open(score_relax, "r") as f:
                data_relax = f.readlines()
            info_list = []
            score_list = []
            for data in data_relax:
                info = json.loads(data)
                score = info[score_key]
                info_list.append(info)
                score_list.append(score)
            min_idx = np.argsort(np.array(score_list))[0]
            
            pdb_relaxed_real_name = info_list[min_idx]['decoy'] + '.pdb'
            pdb_relaxed_real_temp = os.path.join(output_relax_input_path, pdb_relaxed_real_name)
            cmd = 'cp ' + pdb_relaxed_real_temp + ' ' + pdb_relaxed
            os.system(cmd)
    
    # -------------------------------- 2. local dock -------------------------------- #
    print(f'-- 2. local dock --')
    if is_accurate:
        input_file = pdb_relaxed
        pdb_docked_name = pdb_id + '_relaxed_0001.pdb'
        num_struct = 500
    else:
        input_file = os.path.join(workdir, pdb_id + '.pdb')
        pdb_docked_name = pdb_id + '_0001.pdb'
        num_struct = simple_docking_samples
    
    pdb_docked_temp = os.path.join(output_docking_path, pdb_docked_name)
    pdb_docked = os.path.join(workdir, pdb_id + '_docked.pdb')
    score_docking = os.path.join(output_docking_path, 'score.sc')
    
    if not os.path.exists(pdb_docked_temp):
        if is_parallel:
            cmd = 'mpirun -np ' + str(cpu_num) + \
                  ' docking_protocol.mpi.linuxgccrelease -in:file:s ' + input_file + \
                  ' -out:path:all ' + output_docking_path + \
                  ' -scorefile_format json' + \
                  ' -partners ' + chain + \
                  ' -mute all -dock_pert 3 8 -ex1 -ex2aro -nstruct ' + str(num_struct)
        else:
            cmd = 'docking_protocol.linuxgccrelease -in:file:s ' + input_file + \
                  ' -out:path:all ' + output_docking_path + \
                  ' -scorefile_format json' + \
                  ' -partners ' + chain + \
                  ' -mute all -dock_pert 3 8 -ex1 -ex2aro -nstruct ' + str(num_struct)
        os.system(cmd)
        
        with open(score_docking, "r") as f:
            data_relax = f.readlines()
        info_list = []
        score_list = []
        for data in data_relax:
            info = json.loads(data)
            score = info[score_key]
            info_list.append(info)
            score_list.append(score)
        min_idx = np.argsort(np.array(score_list))[0]
        
        pdb_docked_real_name = info_list[min_idx]['decoy'] + '.pdb'
        pdb_docked_real_temp = os.path.join(output_docking_path, pdb_docked_real_name)
        
        cmd = 'cp ' + pdb_docked_real_temp + ' ' + pdb_docked
        os.system(cmd)
        
        if not is_accurate:
            final_score = score_list[min_idx]
            print(f'--finish SIMPLE docking pdb:{pdb_id} !--')
            print(f'final docking score is {final_score}')
    
    if is_accurate:
        # -------------------------------- 3. local refine -------------------------------- #
        print(f'-- 3. local refine --')
        input_file = pdb_docked
        
        pdb_refined_name = pdb_id + '_docked_0001.pdb'
        pdb_refined_temp = os.path.join(output_refine_path, pdb_refined_name)
        pdb_refined = os.path.join(workdir, pdb_id + '_refined.pdb')
        score_refine = os.path.join(output_refine_path, 'score.sc')
        
        if not os.path.exists(pdb_refined_temp):
            if is_parallel:
                cmd = 'mpirun -np ' + str(1) + \
                      ' docking_protocol.mpi.linuxgccrelease -in:file:s ' + input_file + \
                      ' -out:path:all ' + output_refine_path + \
                      ' -scorefile_format json' + \
                      ' -partners ' + chain + \
                      ' -mute all -dock_pert 3 8 -ex1 -ex2aro -nstruct 1  -use_input_sc -docking_local_refine'
            else:
                cmd = 'docking_protocol.linuxgccrelease -in:file:s ' + input_file + \
                      ' -out:path:all ' + output_refine_path + \
                      ' -scorefile_format json' + \
                      ' -partners ' + chain + \
                      ' -mute all -dock_pert 3 8 -ex1 -ex2aro -nstruct 1  -use_input_sc -docking_local_refine'
            os.system(cmd)
            
            cmd = 'cp ' + pdb_refined_temp + ' ' + pdb_refined
            os.system(cmd)
        
        # -------------------------------- 4. output relax -------------------------------- #
        print(f'-- 4. output relax --')
        input_file = pdb_refined
        
        pdb_result_name = pdb_id + '_refined_0001.pdb'
        pdb_result_temp = os.path.join(output_relax_output_path, pdb_result_name)
        pdb_result = os.path.join(workdir, pdb_id + '_result.pdb')
        score_result = os.path.join(output_relax_output_path, 'score.sc')
        
        if not os.path.exists(pdb_result_temp):
            if is_parallel:
                cmd = 'mpirun -np ' + str(2) + \
                      ' relax.mpi.linuxgccrelease -in:file:s ' + input_file + \
                      ' -out:path:all ' + output_relax_output_path + \
                      ' -scorefile_format json' + \
                      ' -nstruct 2 -relax:constrain_relax_to_start_coords -relax:ramp_constraints false -ex1 -ex2 -use_input_sc -flip_HNQ -no_optH false -mute all'
            else:
                cmd = 'relax.default.linuxgccrelease -in:file:s ' + input_file + \
                      ' -out:path:all ' + output_relax_output_path + \
                      ' -scorefile_format json' + \
                      ' -nstruct 1 -relax:constrain_relax_to_start_coords -relax:ramp_constraints false -ex1 -ex2 -use_input_sc -flip_HNQ -no_optH false -mute all'
            os.system(cmd)
            
            with open(score_result, "r") as f:
                data_relax = f.readlines()
            info_list = []
            score_list = []
            for data in data_relax:
                info = json.loads(data)
                score = info['total_score']
                info_list.append(info)
                score_list.append(score)
            min_idx = np.argsort(np.array(score_list))[0]
            
            pdb_result_real_name = info_list[min_idx]['decoy'] + '.pdb'
            pdb_result_real_temp = os.path.join(output_relax_output_path, pdb_result_real_name)
            
            cmd = 'cp ' + pdb_result_real_temp + ' ' + pdb_result
            os.system(cmd)
            
            final_score = score_list[min_idx]
            
            print(f'--finish docking pdb:{pdb_id} !--')
            print(f'final docking score is {final_score}')
    
    return final_score


def docking_script_abbind(pdb_id, pdb_path, is_parallel, docking_folder, chain,
                   is_accurate=False, simple_docking_samples=50, score_key='total_score'):
    """
        -workdir (docking_folder)
            -PDB_id (workdir)
                -PDB_id.pdb
                -(PDB_id_relaxed.pdb)
                -(PDB_id_docked.pdb)
                -(PDB_id_refined.pdb)
                -(PDB_id_result.pdb)
                -(score.sc) (? or in temp)
                -<temp> output_files
                    -output_relax_input: 2
                    -output_docking: 500
                    -output_refine: 1
                    -output_relax_output: 2

        -PDB_id.csv
    """
    workdir = os.path.join(docking_folder, pdb_id)
    if not os.path.exists(workdir):
        os.system('mkdir ' + workdir)
    
    command = 'cp ' + pdb_path + ' ' + workdir
    os.system(command)
    
    clear_dir_single(workdir)
    
    final_score = docking_simple_single_abbind(workdir, pdb_id, chain, is_parallel,
                                        is_accurate=is_accurate, simple_docking_samples=simple_docking_samples, score_key=score_key)
    
    return final_score

