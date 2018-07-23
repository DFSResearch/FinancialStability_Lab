import os, sys
import shutil
from datetime import datetime

files_to_move = ['dataextraction',  'main', 'weight_scheme',]
files_in_folders = [os.path.join('datapreparation', 'prep_functions')]
files_to_move.extend(sys.argv[1:])

current_day = datetime.now().day
current_month = datetime.now().month
folder_archive = os.path.join('Archive', f'{current_day}_{current_month}')

if not os.path.exists(folder_archive):
    os.makedirs(folder_archive)

def file_renamer(file_name, format_='.py'):
    new_name = os.path.join(folder_archive, ''.join([file_name, 
                    f'_copy_{current_day}_{current_month}{format_}']))
    return new_name

print(list(map(file_renamer, files_to_move)))
                            

for file in files_to_move:
    if file == 'main':
        shutil.copyfile(file + '.ipynb', file_renamer(file, '.ipynb'))
    else:
        shutil.copyfile(file + '.py', file_renamer(file))
                            
for full_path in files_in_folders:
    path_array = os.path.split(full_path)
    folder = path_array[0]
    file_itself = path_array[1]
    new_folder = file_renamer(folder)
    os.mkdir(new_folder)
    shutil.copyfile(full_path + '.py', os.path.join(new_folder, file_itself + '.py'))               