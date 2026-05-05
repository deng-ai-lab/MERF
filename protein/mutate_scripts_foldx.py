import os

# def fix_structure(filepath):
#     os.system('./profix -fix 0 ' + '/' + filepath + '/complex.pdb')
#     os.system('mv complex_fix.pdb complex.pdb')
#     os.system('mv complex.pdb ' + filepath + '/complex.pdb')
#     return

def Repair_wt(pdb_id, wt_path, fix_path):
    workdir = f'{pdb_id}_repair'
    if os.path.exists(workdir):
        os.system('rm -r ' + workdir)
    os.system('mkdir ' + workdir)
    os.system('cp ' + wt_path + ' ' + workdir)

    os.chdir(workdir)

    command = f"foldx_20251231 --command=RepairPDB --pdb={pdb_id}.pdb"
    os.system(command)

    os.system('mv ' + pdb_id + '_Repair.pdb ' + fix_path)

    os.chdir('..')
    os.system('rm -r ' + workdir)

def Gen_mut(pdb_id, fix_path, mutate_info, mut_path):
    workdir = f'{pdb_id}_{mutate_info}_mutate'
    if os.path.exists(workdir):
        os.system('rm -r ' + workdir)
    os.system('mkdir ' + workdir)
    os.system('cp ' + fix_path + ' ' + workdir)

    os.chdir(workdir)

    # write the mutate_info into a txt file
    mutate_file = 'individual_list.txt'
    with open(mutate_file, 'w') as f:
        f.write(mutate_info + ';\n')
    
    command = f"foldx_20251231 --command=BuildModel --pdb={pdb_id}.pdb --mutant-file={mutate_file}"
    os.system(command)

    os.system('mv ' + pdb_id + '_1.pdb ' + mut_path)

    os.chdir('..')
    os.system('rm -r ' + workdir)

def mut_list(pdb_id, mutate_list, wt_dir, fix_dir, mut_dir):
    # 需要在外面处理好mutate_list的格式，为list，逗号分割多点突变。old_mutate_infos专用于abbind数据集，保存原始突变信息以命名文件
    os.chdir('protein')

    wt_path = os.path.join(wt_dir, pdb_id + '.pdb')
    fix_path = os.path.join(fix_dir, pdb_id + '.pdb')
    
    # 根据wt路径是否存在，判断是否需要修复wt结构
    if not os.path.exists(fix_path):
        Repair_wt(pdb_id, wt_path, fix_path)

    # 根据mutate_list逐个判断突变文件是否存在，若不存在则进行突变
    for mutate_info in mutate_list:
        mut_path = os.path.join(mut_dir, pdb_id + '_' + mutate_info + '.pdb')
        
        if os.path.exists(mut_path):
            continue
        
        # 进行突变
        Gen_mut(pdb_id, fix_path, mutate_info, mut_path)
        
    os.chdir('..')