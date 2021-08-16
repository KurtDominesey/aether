import os

import numpy as np

import paraview.simple as pv


pv._DisableFirstRenderCameraReset()


def plot_spatial_view(filename, savebase):
    # This script was written for both version 4.4 and 5.9 of ParaView.
    # This is because 4.4 is the last version to support exporting
    # unrasterized geometry in vector graphics.
    # Unfortunately, bugs remain in 4.4. Namely, the colorbar label font size
    # appears to do nothing above a certain value (fixed in 5.4). Also, ParaView
    # crashes during the export after saving the image as a temporary (pvtmp)
    # file.
    # For this reason, the script is only completely usable with ParaView 5.9.
    version = pv.GetParaViewVersion().GetVersion()
    if version <= 4.4:
        reader = pv.XMLUnstructuredGridReader(FileName=filename+'.vtu')
    else:
        reader = pv.OpenDataFile(filename+'.vtu')
    pv.SetActiveSource(reader)
    view = pv.GetActiveViewOrCreate('RenderView')
    if version >= 5.9:
        view.ViewSize = [660, 380]
    else:
        view.ViewSize = [660, 400]
    layout = pv.GetLayout()
    if version <= 4.4:
        display = pv.Show(reader, view)
        view.InteractionMode = '2D'
    else:
        display = pv.Show(reader, view, 'UnstructuredGridRepresentation')
    display.Representation = 'Surface'
    view.ResetCamera()
    camera = pv.GetActiveCamera()
    x, y, z = camera.GetPosition()
    if version >= 5.9:
        move = 1.75
    else:
        move = 1.65
    camera.SetPosition(x*move, y, z)
    x, y, z = camera.GetFocalPoint()
    camera.SetFocalPoint(x*move, y, z)
    if version > 4.4:
        camera.Dolly(1.45)  # magic number to zoom into plot
    else:
        view.CameraParallelProjection = 1
        view.CameraParallelScale = 0.316  # magic number to zoom into plot
    view.OrientationAxesVisibility = 0
    view.Background = (1, 1, 1)
    for field in reader.PointData.keys():
        pv.ColorBy(display, ('POINTS', field))
        if version > 4.4:
            pv.AssignLookupTable(reader.PointData[field], 
                                 'Inferno (matplotlib)')
            display.RescaleTransferFunctionToDataRange(True, False)
        else:
            display.RescaleTransferFunctionToDataRange(True)
        display.SetScalarBarVisibility(view, False)
        colorfunc = pv.GetColorTransferFunction(field)
        if version <= 4.4:
            colorfunc.ApplyPreset('Black-Body Radiation', True)
        colorbar = pv.GetScalarBar(colorfunc, view)
        colorbar.AutomaticLabelFormat = False
        colorbar.LabelFormat = '%-#6.2e'
        colorbar.Title = ''
        colorbar.ComponentTitle = ''
        colorbar.LabelFontFamily = 'Times'
        if version >= 5.9:
            colorbar.LabelFontSize = 54
            colorbar.ScalarBarThickness = 24
        else:
            colorbar.LabelFontSize = 40  # this does nothing in version 4.4 (?)
        colorbar.LabelColor = (0, 0, 0)
        if version > 4.4:
            colorbar.ScalarBarLength = 0.85
            colorbar.UseCustomLabels = True
            dmin,  dmax = reader.PointData[field].GetRange()
            colorbar.CustomLabels = \
                np.linspace(dmin, dmax, 4, endpoint=False)[1:]
        else:
            colorbar.NumberOfLabels = 5
            colorbar.Position2[1] = 0.88
            print(colorbar.Position, colorbar.Position2)
        end = '-'.join(field.split('_')[-2:])
        pv.Show()
        pv.Render()
        savename = 'views/{savebase}-{end}'.format(savebase=savebase, end=end)
        if version <= 4.4:
            display.InterpolateScalarsBeforeMapping = False
            display.BackfaceRepresentation = 'Cull Backface'
            pv.ExportView('views/foobar4.pdf', view, Rasterize3Dgeometry=False)
        else:
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
    # "C:\Program Files\ParaView 5.9.1-Windows-Python3.8-msvc2017-64bit\bin\pvpython" plot_spatial_views.py
    base = 'GroupStructure_CathalauCompareTestWithUpdate' \
           '_{fuel}_{structure}_{algorithm}_error'
    fuels = ['uo2', 'mox43']
    structures = ['CASMO-70', 'XMAS-172', 'SHEM-361']
    algorithms = ['pgd']  # , 'svd']
    for fuel in fuels:
        for structure in structures:
            for algorithm in algorithms:
                filename = base.format(fuel=fuel, structure=structure, 
                                       algorithm=algorithm)
                groups = structure.split('-')[-1]
                savebase = '{fuel}-{groups}-{algorithm}'.format(
                    fuel=fuel, groups=groups, algorithm=algorithm)
                plot_spatial_view(filename, savebase)
