import re
import sys

import h5py
import numpy as np

def compare_mgxs(filename_a, filename_b, savename):
    file_a = h5py.File(filename_a, 'r')
    file_b = h5py.File(filename_b, 'r')
    file_d = h5py.File(savename, 'w')
    file_d.attrs['energy_groups'] = file_a.attrs['energy_groups']
    for name, material_a in file_a.items():
        if not isinstance(material_a, h5py.Group) or name == 'void':
            continue
        material_b = file_b[name]
        material_d = file_d.create_group(name)
        for temperature in material_a:
            if not re.match('[0-9]+K', temperature):
                continue
            total_a = np.array(material_a[temperature]['total'])
            total_b = np.array(material_b[temperature]['total'])
            library = material_d.create_group(temperature)
            total_d = library.create_dataset('total', total_a.shape)
            total_d[:] = (total_a - total_b)
    file_a.close()
    file_b.close()
    file_d.close()

if __name__ == '__main__':
    compare_mgxs(*sys.argv[1:])

