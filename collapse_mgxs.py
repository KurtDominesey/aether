import re
import sys

import h5py

def collapse_mgxs(filename, savename, g_maxes):
    file_fine = h5py.File(filename, 'r')
    file_coarse = h5py.File(savename, 'w')
    group_structure = file_fine.attrs['group structure']
    group_widths = group_structure[1:] - group_structure[:-1]
    file_coarse.attrs['energy_groups'] = len(g_maxes)
    for name, material in file_fine.items():
        if not isinstance(material, h5py.Group):
            continue
        material_coarse = file_coarse.create_group(name)
        for temperature in material:
            if not re.match('[0-9]+K', temperature):
                continue
            library = material_coarse.create_group(temperature)
            library.create_dataset('total', (len(g_maxes),))
    for gc, g_max in enumerate(g_maxes):
        g_min = g_maxes[gc-1] if gc > 0 else 0
        subgroups = range(g_min, g_max)
        group_width = sum(group_widths[g] for g in subgroups)
        for name, material in file_fine.items():
            totals = material['294K']['total']
            total = sum(totals[g] * group_widths[g] for g in subgroups)
            total /= group_width
            file_coarse[name]['294K']['total'][gc] = total
    file_fine.close()
    file_coarse.close()

if __name__ == '__main__':
    g_maxes = (10, 14, 18, 24, 43, 55, 65, 70)
    collapse_mgxs(sys.argv[1], sys.argv[2], g_maxes)

