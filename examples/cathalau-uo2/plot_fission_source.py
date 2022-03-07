import os

import numpy as np

import paraview.simple as pv


pv._DisableFirstRenderCameraReset()


def plot_fission_source(filename, savebase):
    # This script was written for both version 5.9 of ParaView.
    reader = pv.OpenDataFile(filename+'.vtu')
    pv.SetActiveSource(reader)
    view = pv.GetActiveViewOrCreate('RenderView')
    view.ViewSize = [1000, 700] #[660*2+330, 380*2+190]
    layout = pv.GetLayout()
    display = pv.Show(reader, view, 'UnstructuredGridRepresentation')
    display.Representation = 'Surface'
    view.ResetCamera()
    camera = pv.GetActiveCamera()
    x, y, z = camera.GetPosition()
    move_x = 0.99
    move_y = 0.69
    camera.SetPosition(x*move_x, y*move_y, z)
    x, y, z = camera.GetFocalPoint()
    camera.SetFocalPoint(x*move_x, y*move_y, z)
    camera.Dolly(2.1)  # magic number to zoom into plot
    view.OrientationAxesVisibility = 0
    view.Background = (1, 1, 1)
    for field in reader.PointData.keys():
        if field == 0 or field == 1 or (field == 'ref' and 'Minimax' in savebase):
            continue
        pv.ColorBy(display, ('POINTS', field))
        pv.AssignLookupTable(reader.PointData[field], 'Inferno (matplotlib)')
        field_range = reader.PointData[field].GetRange()
        print(field_range)
        display.RescaleTransferFunctionToDataRange(True, False)
        display.SetScalarBarVisibility(view, False)
        colorfunc = pv.GetColorTransferFunction(field)
        colorfunc.NanOpacity = 0
        colorbar = pv.GetScalarBar(colorfunc, view)
        colorbar.AutomaticLabelFormat = False #!
        if field == 'ref':
            colorbar.LabelFormat = '%-#6.2f'
        else:
            colorbar.LabelFormat = '%-#6.1e'
        colorbar.RangeLabelFormat = '%-#6.2f'
        colorbar.AddRangeLabels = False
        colorbar.Title = ''
        colorbar.ComponentTitle = ''
        colorbar.LabelFontFamily = 'Times'
        if field == 'ref':
            colorbar.LabelFontSize = 85
            colorbar.ScalarBarThickness = 35
        else:
            colorbar.LabelFontSize = 75
            colorbar.ScalarBarThickness = 25
        colorbar.LabelColor = (0, 0, 0)
        colorbar.ScalarBarLength = 0.85
        colorbar.UseCustomLabels = True #!
        # dmin,  dmax = reader.PointData[field].GetRange()
        # if 'd' in field:
        #     dmin, _ = reader.PointData[field.replace('d', 'm')].GetRange()
        # else:
        #     _, dmax = reader.PointData[field.replace('m', 'd')].GetRange()
        # colorfunc.RescaleTransferFunction(dmin, dmax)
        # colorbar.CustomLabels = \
        #     np.linspace(dmin, dmax, 4, endpoint=False)[1:]
        if field == 'ref':
            drange = (0.8765203952789307, 1.1870371103286743)
            colorbar.CustomLabels = [drange[0], 0.95, 1, 1.05, 1.1, 1.15, drange[1]]
            colorfunc.RescaleTransferFunction(*drange)
        else:
            colorbar.CustomLabels = np.linspace(*field_range, 5, endpoint=True)
            colorfunc.RescaleTransferFunction(*field_range)
        end = '-'.join(field.split('_')[-2:])
        pv.Show()
        pv.Render()
        savename = 'fiss-src/{savebase}-{end}'.format(savebase=savebase, end=end)
        pv.ExportView(savename+'.pdf', view, Compressoutputfile=True)
        os.replace(savename+'.pdf.pvtmp.pdf', savename+'.pdf')
        # scale = 1  # integers only
        # res = [int(view.ViewSize[i]*scale) for i in range(2)]
        # pv.SaveScreenshot(savename+'.png', view, TransparentBackground=True,
        #                   CompressionLevel=9, ImageResolution=res)
        pv.Delete(colorbar)


if __name__ == '__main__':
    # The easiest way to run this script is with pvpython.
    # See the bin folder of your ParaView installation.
    # pvpython plot_spatial_views.py
    # "C:\Program Files\ParaView 5.9.1-Windows-Python3.8-msvc2017-64bit\bin\pvpython" plot_fission_source.py
    base = 'GroupStructure_CathalauCompareTest{algorithm}WithEigenUpdate' \
           '_{fuel}_{structure}_diff_q'
    fuels = ['uo2', 'mox43']
    structures = ['CASMO-70', 'XMAS-172', 'SHEM-361']
    algorithms = ['', 'Minimax']
    for fuel in fuels:
        for structure in structures:
            for algorithm in algorithms:
                filename = base.format(fuel=fuel, structure=structure, 
                                       algorithm=algorithm)
                groups = structure.split('-')[-1]
                savebase = '{fuel}-{groups}-{algorithm}'.format(
                    fuel=fuel, groups=groups, algorithm=algorithm)
                plot_fission_source(filename, savebase)