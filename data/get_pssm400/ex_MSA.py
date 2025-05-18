import os

# Proteins are saved in fasta format, and instructions for generating pssm files using psiblast are provided here
input = '/home/ywh/blast/bin/fasta_data'
tmp = '/home/ywh/blast/bin/tmp'
pssm = '/home/ywh/blast/bin/pssm'
while not len(os.listdir(input)) == 0:
    # 获取文件的完整路径
    file_list = os.listdir(input)
    file = file_list[0]
    name = file.split('.')[0]
    file_path = os.path.join(input, file)
    out_path = os.path.join(tmp, name+'.pssm')
    os.system('/home/ywh/blast/bin/psiblast -query '+file_path+' -db /home/ywh/blast/bin/nr -out_ascii_pssm '+out_path+' -num_threads 16 -num_iterations 3 -evalue 0.001')
    os.system('cp '+out_path+' '+pssm)
    os.system('rm -f '+out_path)
    os.system('rm -f ' + file_path)
