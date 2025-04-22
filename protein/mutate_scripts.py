import os

# Generate Mutant structure using scap
def Gen_mut(workdir, pdbid, chainid, resid, mutname, idx):
    scapfile = open('./tmp_scap.list', 'w')
    scapfile.write(chainid + ',' + str(resid) + ',' + mutname)
    scapfile.close()
    if idx == 0:
        filename = workdir + '/' + pdbid + '.pdb'
    else:
        filename = workdir + '/' + pdbid + '_mut.pdb'
    os.system('./scap -ini 20 -min 4 ' + filename + ' ./tmp_scap.list')
    
    # change the mutation file to shorter name
    tmp_name = workdir + '/' + 'tmp.pdb'
    cmd = 'mv ' + filename + ' ' + tmp_name
    os.system(cmd)
    
    os.system('./scap -ini 20 -min 4 ' + tmp_name + ' ./tmp_scap.list')
    
    os.system('mv ' + 'tmp_scap.pdb ' + workdir + '/' + pdbid + '_mut.pdb')

    os.system('rm ./tmp_scap.list')
    return


def modifypdb(name1, name2):
    pdbfile = open(name1)
    outfile = open(name2, "w")
    lines = pdbfile.read().splitlines()
    PRO = []
    for line in lines:
        if line[0:4] == 'ATOM':
            outfile.write(line + '\n')
    outfile.close()
    return


def fix_structure(filepath):
    os.system('./profix -fix 0 ' + '/' + filepath + '/complex.pdb')  # 调用data中software
    os.system('mv complex_fix.pdb complex.pdb')
    os.system('mv complex.pdb ' + filepath + '/complex.pdb')
    return


def mut_list_abbind(pdb_id, pdb_path, mut_path, mutate_info_list, chainid):
    os.chdir('protein')
    
    for idx in range(len(mutate_info_list)):
        mutate_info = mutate_info_list[idx]
        
        PDB_wt_file_path = pdb_path
        PDB_mut_name = pdb_id + '_' + mutate_info + '.pdb'
        
        target_dir = '../data/ABbind/PDBs_evo_3'
        mutated_files = os.listdir(target_dir)
        
        # the first time to delete repeat
        if PDB_mut_name in mutated_files:
            continue
        
        # mutate
        # create the workdir
        workdir = 'pdbfile'
        if os.path.exists(workdir):
            os.system('rm -r pdbfile')
        if not os.path.exists(workdir):
            os.system('mkdir ' + workdir)
        command = 'cp ' + PDB_wt_file_path + ' ' + workdir
        os.system(command)
        
        # fix structure, no need here, already fixed
        # modifypdb(workdir + '/' + pdb_id + '.pdb', workdir + '/complex.pdb')
        # fix_structure(workdir)
        # os.system("mv " + workdir + '/complex.pdb ' + workdir + '/' + pdb_id_iter + '.pdb ')
        # os.system()
        
        # mutate files
        mutate_list = mutate_info.split('_')
        
        # temp_mutate_info = ''
        for i, single_mutate in enumerate(mutate_list):
            
            resid = str(int(single_mutate[1:-1]))
            mutname = single_mutate[-1]
            
            Gen_mut(workdir, pdb_id, chainid, str(resid), mutname, i)
            
        os.system('mv ' + workdir + '/' + pdb_id + '_mut.pdb ' + mut_path)
    
    os.chdir('..')
