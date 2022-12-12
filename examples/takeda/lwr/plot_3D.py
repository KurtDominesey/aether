import os
import math

import numpy as np
from matplotlib.ticker import AutoLocator

import paraview.simple as pv


pv._DisableFirstRenderCameraReset()


def plot_3D(filename, viewsize, groups, model):
    # This script was written for version 5.9 of ParaView.
    pv.ResetSession()
    reader = pv.OpenDataFile(filename+'.vtu')
    pv.SetActiveSource(reader)
    view = pv.GetActiveViewOrCreate('RenderView')
    view.ViewSize = viewsize
    # camera_state = (view.CameraPosition,
    #                 view.CameraFocalPoint,
    #                 view.CameraParallelScale)
    if model == 'Lwr':
        view.CameraPosition = [-29.325815186890406, -67.31404032764105, 53.29753603314557]
        view.CameraFocalPoint = [12.499999999999998, -4.57531754730548, 17.075317547305485]
        view.CameraViewUp = [0.22414386804201333, 0.3708909791235276, 0.9012210650134381]
        view.CameraParallelScale = 21.650635094610966
        view.CameraParallelProjection = 1
    elif 'Pin' in model:
      view.CameraPosition = [-10.461710363043647, 18.980809880731925, 18.86987337366635]
      view.CameraFocalPoint = [0.3149999976158133, 0.3149999976158133, 6.426000118255615]
      view.CameraViewUp = [0.24999999999999994, -0.4330127018922193, 0.8660254037844387]
      # view.CameraPosition = [70.88550620819066, -78.69742850340742, 96.86591356945476]
      # view.CameraFocalPoint = [0.3149999976158133, 0.3149999976158133, 32.130001068115234]
      # view.CameraViewUp = [-0.35451329085019884, 0.3824506249672406, 0.8532595420343856]
      view.CameraParallelScale = 6.418084298326854
      view.CameraParallelProjection = 1

      scale_z = 0.2
      transform = pv.Transform(registrationName='Transform', Input=reader)
      transform.Transform = 'Transform'
      transform.Transform.Scale = (1, 1, scale_z)
      reader = transform

      nudge = 1e-6
      diag = 0.54 / math.sqrt(2)
      circle = pv.Ellipse(registrationName='Circle')
      circle.Center = (0, 0, 21.42*scale_z+nudge)
      circle.MajorRadiusVector = (0.54, 0, 0)
      circle.SectorAngle = 90
      circle.Close = False

      edge_y = pv.Line(registrationName='EdgeY')
      edge_y.Point1 = (0, 0.54, 0)
      edge_y.Point2 = (0, 0.54, 21.42*scale_z+nudge)

      edge_d = pv.Line(registrationName='EdgeD')
      edge_d.Point1 = (diag, diag, 21.42*scale_z+nudge)
      edge_d.Point2 = (diag, diag, 21.42*2*scale_z)

      edge_z = pv.Line(registrationName='EdgeZ')
      edge_z.Point1 = edge_d.Point2
      edge_z.Point2 = (0, 0, edge_z.Point1[2])

      outline = [edge_y, circle, edge_d, edge_z]
      for path in outline:
        display = pv.Show(path, view, 'UnstructuredGridRepresentation')
        display.Representation = 'Surface'
        display.LineWidth = 6
        display.AmbientColor = (0, 1, 0)
        display.DiffuseColor = (0, 1, 0)
    camera = pv.GetActiveCamera()
    # camera.Elevation(elevation)
    # camera.Azimuth(azimuth)
    # camera.Roll(roll)
    # view.OrientationAxesVisibility = 0
    # setup_pv()
    display = pv.Show(reader, view, 'UnstructuredGridRepresentation')
    display.Representation = 'Surface'
    # display.ColorArrayName = (None, '')
    view.UseColorPaletteForBackground = False
    view.Background = (1,1,1)
    view.OrientationAxesVisibility = False
    view.OrientationAxesLabelColor = (0,0,0)

    ticker = AutoLocator()

    # clip stuff
    origin = (12.5, 12.5, 15) if model == 'Lwr' else (0, 0, 21.42*scale_z)

    clip_z = pv.Clip(registrationName='ClipZ', Input=reader)
    clip_z.ClipType = 'Plane'
    clip_z.HyperTreeGridClipper = 'Plane'
    clip_z.ClipType.Origin = origin
    clip_z.ClipType.Normal = (0, 0, 1)
    clip_z_display = pv.Show(clip_z, view, 'UnstructuredGridRepresentation')
    
    clip_d = pv.Clip(registrationName='ClipD', Input=reader)
    clip_d.ClipType = 'Plane'
    clip_d.HyperTreeGridClipper = 'Plane'
    clip_d.ClipType.Origin = origin
    clip_d.ClipType.Normal = (1, -1, 0) if model == 'Lwr' else (-1, 1, 0)
    clip_d_display = pv.Show(clip_d, view, 'UnstructuredGridRepresentation')
    
    # hide original
    pv.Hide(reader, view)
    for group in groups:
        field = 'g' + str(group)
        pv.AssignLookupTable(reader.PointData[field], 'Inferno (matplotlib)')
        field_range = reader.PointData[field].GetRange()
        # fmin, fmax = field_range
        # print(field_range)
        # display.SetScaleArray = ('Points', field)
        # display.ScaleTransferFunction = 'PiecewiseFunction'
        # display.ScaleTransferFunction.Points = (fmin, 0, 0.5, 0, 
        #                                         fmax, 1, 0.5, 0)
        # display.OpacityArray = ('Points', field)
        # display.OpacityTransferFunction = 'PiecewiseFunction'
        # display.OpacityTransferFunction.Points = (fmin, 0, 0.5, 0, 
        #                                           fmax, 1, 0.5, 0)
        # display.OpacityArrayName = ('Points', field)
        pv.ColorBy(display, ('Points', field))
        display.RescaleTransferFunctionToDataRange(True, False)
        pv.ColorBy(clip_z_display, ('Points', field))
        pv.ColorBy(clip_d_display, ('Points', field))
        # display.SetScalarBarVisibility(view, False)
        colorfunc = pv.GetColorTransferFunction(field)
        # # colorfunc.NanOpacity = 0
        colorbar = pv.GetScalarBar(colorfunc, view)
        colorbar.AutomaticLabelFormat = False #!
        # if field == 'ref':
        colorbar.LabelFormat = '%-#6.2f'
        # else:
        # colorbar.LabelFormat = '%-#6.1e'
        colorbar.RangeLabelFormat = '%-#6.2f'
        # colorbar.AddRangeLabels = False
        colorbar.Title = ''
        colorbar.ComponentTitle = ''
        colorbar.LabelFontFamily = 'Times'
        colorbar.LabelFontSize = 48 if model == 'Lwr' else 64
        colorbar.ScalarBarThickness = 20 if model == 'Lwr' else 40
        colorbar.LabelColor = (0, 0, 0)
        colorbar.ScalarBarLength = 0.93
        colorbar.UseCustomLabels = True #!
        colorbar.CustomLabels = ticker.tick_values(*field_range)
        # fmin,  fmax = reader.PointData[field].GetRange()
        # # if 'd' in field:
        # #     dmin, _ = reader.PointData[field.replace('d', 'm')].GetRange()
        # # else:
        # #     _, dmax = reader.PointData[field.replace('m', 'd')].GetRange()
        # # colorfunc.RescaleTransferFunction(dmin, dmax)
        # # colorbar.CustomLabels = \
        # #     np.linspace(dmin, dmax, 4, endpoint=False)[1:]
        # colorbar.CustomLabels = np.linspace(*field_range, 7, endpoint=True)
        # # colorfunc.RescaleTransferFunction(*field_range)
        pv.ResetCamera()
        # camera.Dolly(3)
        zoom = 0.94 if model == 'Lwr' else 0.275
        view.CameraParallelScale = zoom * view.CameraParallelScale
        x, y, z = camera.GetPosition()
        move_x = 3.8 if model == 'Lwr' else -0.75
        move_y = 1.2 if model == 'Lwr' else 0.5
        camera.SetPosition(x+move_x, y-move_y, z)
        x, y, z = camera.GetFocalPoint()
        camera.SetFocalPoint(x+move_x, y-move_y, z)
        pv.Show()
        pv.Render()
        # pv.RenderAllViews()
        savename = f'{filename}-{field}.pdf'
        pv.ExportView(savename, view, Compressoutputfile=True, 
            Rasterize3Dgeometry=False)
        # os.replace(savename+'.pdf.pvtmp.pdf', savename+'.pdf')
        # scale = 1  # integers only
        # res = [int(view.ViewSize[i]*scale) for i in range(2)]
        # pv.SaveScreenshot(savename+'.png', view, TransparentBackground=True,
        #                   CompressionLevel=9, ImageResolution=res)
        pv.Delete(colorbar)
    # view.CameraPosition, view.CameraFocalPoint, view.CameraParallelScale = \
    #     camera_state
    # camera.Elevation(-elevation)
    # camera.Azimuth(-azimuth)
    # camera.Roll(-roll)
    pv.Delete(reader)
    pv.Delete(clip_z)
    pv.Delete(clip_d)
    


def setup_pv_lwr():
    camera = pv.GetActiveCamera()
    camera.Azimuth(27.368+180)
    camera.Elevation(31.174)
    # x, y, z = camera.GetPosition()
    # move_x = 0.99
    # move_y = 0.69
    # camera.SetPosition(x*move_x, y*move_y, z)
    # x, y, z = camera.GetFocalPoint()
    # camera.SetFocalPoint(x*move_x, y*move_y, z)
    # camera.Dolly(2.1)  # magic number to zoom into plot

def setup_pv_pin():
    pass


if __name__ == '__main__':
    # The easiest way to run this script is with pvpython.
    # See the bin folder of your ParaView installation.
    # pvbatch plot_3D.py
    # "C:\Program Files\ParaView 5.10.0-Windows-Python3.9-msvc2017-AMD64\bin\pvbatch.exe" plot_3D.py
    filespec = 'out/BenchmarksTest3DFixedSource{model}{case}'
    kwargs_lwr = dict(groups=(1, 2), viewsize=(1100, 1000))
                      # elevation=-60, azimuth=-30, roll=0)
                      # azimuth=-27.368, elevation=31.174-90, roll=0)
    kwargs_pin = dict(groups=(1, 6, 7), viewsize=(600, 2*1000))
    runs = {
      ('Lwr', ('RodY', 'RodN')): kwargs_lwr,
      ('PinUO2', ('Hgt2',)): kwargs_pin
    }
    for ((model, cases), kwargs) in runs.items():
      for case in cases:
        filename = filespec.format(model=model, case=case)
        plot_3D(filename, model=model, **kwargs)