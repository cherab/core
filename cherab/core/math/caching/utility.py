# Copyright 2016-2018 Euratom
# Copyright 2016-2018 United Kingdom Atomic Energy Authority
# Copyright 2016-2018 Centro de Investigaciones Energéticas, Medioambientales y Tecnológicas
#
# Licensed under the EUPL, Version 1.1 or – as soon they will be approved by the
# European Commission - subsequent versions of the EUPL (the "Licence");
# You may not use this work except in compliance with the Licence.
# You may obtain a copy of the Licence at:
#
# https://joinup.ec.europa.eu/software/page/eupl5
#
# Unless required by applicable law or agreed to in writing, software distributed
# under the Licence is distributed on an "AS IS" basis, WITHOUT WARRANTIES OR
# CONDITIONS OF ANY KIND, either express or implied.
#
# See the Licence for the specific language governing permissions and limitations
# under the Licence.

import numpy as np
import matplotlib.pyplot as plt

from core.math.caching import Caching2D


def auto_caching2d_optimiser(function2d, space_area, threshold):
    """
    Find the biggest resolution of the caching of function2d allowing a mean
    relative error less than a threshold.
    Also plot a graph with the resolutions tested and their errors.
    :param function2d: the 2D function to cache
    :param space_area: area where the function has to be cached: (minx, maxx, miny, maxy)
    :param threshold: maximum mean relative error wanted
    :return: a tuple of resolutions (resolutionx, resolutiony)
    """

    minx, maxx, miny, maxy = space_area
    nb_samplesx = 50
    nb_samplesy = 50

    current_error = float('inf')
    to_plot_errors = []
    resolutionx = maxx - minx
    resolutiony = maxy - miny
    to_plot_resx = []
    to_plot_resy = []

    while current_error >= threshold:

        resolutionx /= 2
        resolutiony /= 2
        to_plot_resx.append(resolutionx)
        to_plot_resy.append(resolutiony)
        cached_function = Caching2D(function2d, space_area, (resolutionx, resolutiony))
        current_error = 0.
        nb_zeros = 0
        for x in np.linspace(minx, maxx, nb_samplesx):
            for y in np.linspace(miny, maxy, nb_samplesy):
                ref_value = function2d(x, y)
                if ref_value != 0:
                    current_error += abs((cached_function(x, y) - ref_value) / ref_value)
                else:
                    nb_zeros += 1
        current_error /= nb_samplesx * nb_samplesy - nb_zeros
        to_plot_errors.append(current_error)

    plt.plot(to_plot_resx, to_plot_errors, 'o-')
    plt.plot(to_plot_resy, to_plot_errors, 'o-')
    plt.legend(['x resolution', 'y resolution'])
    plt.xlabel('resolution')
    plt.ylabel('relative error')
    plt.xscale('log')
    plt.show()

    return resolutionx, resolutiony


def mapping_caching2d_resolution(function2d, space_area):
    """
    Plot a map of the mean relative error when caching function2d with different resolutions.
    :param function2d: 2D function to be cached
    :param space_area: area where the function has to be cached: (minx, maxx, miny, maxy)
    """

    minx, maxx, miny, maxy = space_area
    nb_samplesx = 50
    nb_samplesy = 50

    errors = np.empty((20, 20))
    resolutionsx = np.logspace(np.log10(maxx - minx) - 2, np.log10(maxx - minx), 20)
    resolutionsy = np.logspace(np.log10(maxy - miny) - 2, np.log10(maxy - miny), 20)

    for i in range(20):
        for j in range(20):

            print(i, j)
            resolutionx = resolutionsx[i]
            resolutiony = resolutionsy[j]
            cached_function = Caching2D(function2d, space_area, (resolutionx, resolutiony))
            error = 0.
            nb_zeros = 0
            for x in np.linspace(minx, maxx, nb_samplesx):
                for y in np.linspace(miny, maxy, nb_samplesy):
                    ref_value = function2d(x, y)
                    if ref_value != 0:
                        error += abs((cached_function(x, y) - ref_value) / ref_value)
                    else:
                        nb_zeros += 1
            error /= nb_samplesx * nb_samplesy - nb_zeros
            errors[i, j] = error

    plt.contourf(resolutionsx, resolutionsy, errors.T)
    plt.xlabel('resolution x')
    plt.ylabel('resolution y')
    plt.xscale('log')
    plt.yscale('log')
    plt.colorbar()
    plt.show()