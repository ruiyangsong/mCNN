import os,shutil
# dataset_name,radii = sys.argv[1:]
dataset_name = 'SSD'
radii = '1.4'

def rm_ref_main():
    # ref_pdb_dir = '/public/home/sry/mCNN/dataset/%s/feature/rosetta/ref_output'%dataset_name
    ref_pdb_dir = r'E:\projects\mCNN\yanglab\dataset\SSD\feature\rosetta\ref_output'
    pdbid_lst = os.listdir(ref_pdb_dir)
    for pdbid in pdbid_lst:
        pthlst = os.listdir(ref_pdb_dir + '/' + pdbid)
        pthlst.remove('%s_ref_radii_1.4.surface.npz'%pdbid)
        for pth in pthlst:
            try:
                os.remove(ref_pdb_dir + '/' + pdbid + '/' + pth)
            except:
                shutil.rmtree(ref_pdb_dir + '/' + pdbid + '/' + pth)

def rm_mut_main():
    # mut_pdb_dir = '/public/home/sry/mCNN/dataset/%s/feature/rosetta/mut_output'%dataset_name
    mut_pdb_dir = r'E:\projects\mCNN\yanglab\dataset\SSD\feature\rosetta\mut_output'
    tag_lst = os.listdir(mut_pdb_dir)
    for tag in tag_lst:
        pdbid = tag.split('_')[0]
        pthlst = os.listdir(mut_pdb_dir + '/' + tag)
        pthlst.remove('%s_mut_radii_1.4.surface.npz'%pdbid)
        for pth in pthlst:
            try:
                os.remove(mut_pdb_dir + '/' + tag + '/' + pth)
            except:
                shutil.rmtree(mut_pdb_dir + '/' + tag + '/' + pth)

def ref_main():
    ref_errlst = []
    # ref_pdb_dir = '/public/home/sry/mCNN/dataset/%s/feature/rosetta/ref_output'%dataset_name
    ref_pdb_dir = 'E:/projects/mCNN/yanglab/dataset/SSD/feature/rosetta/ref_output_surface'
    pdbid_lst = os.listdir(ref_pdb_dir)
    for pdbid in pdbid_lst:
        pth = ref_pdb_dir + '/' + pdbid + '/' + pdbid + '_ref_radii_' + radii + '.surface.npz'
        try:
            assert os.path.exists(pth)
        except:
            ref_errlst.append(pth)
    print(ref_errlst)

def mut_main():
    mut_err_lst = []
    # mut_pdb_dir = '/public/home/sry/mCNN/dataset/%s/feature/rosetta/mut_output'%dataset_name
    mut_pdb_dir = 'E:/projects/mCNN/yanglab/dataset/SSD/feature/rosetta/mut_output_surface'
    tag_lst = os.listdir(mut_pdb_dir)
    for tag in tag_lst:
        pdbid = tag.split('_')[0]
        pth = mut_pdb_dir + '/' + tag + '/' + pdbid + '_mut_radii_' + radii + '.surface.npz'
        try:
            assert os.path.exists(pth)
        except:
            mut_err_lst.append(pth)
    print(mut_err_lst)
if __name__ == '__main__':
    ## remove anything but [pdbid]_ref_radii_1.4.surface.npz
    # rm_ref_main()
    # rm_mut_main()

    ## chk if [pdbid]_ref_radii_1.4.surface.npz exists
    ref_main()
    mut_main()
