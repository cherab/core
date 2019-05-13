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
cimport numpy as np

cpdef tuple sample1d(object function1d, tuple x_range)
cpdef np.ndarray sample1d_points(object function1d, object x_points)

cpdef tuple sample2d(object function2d, tuple x_range, tuple y_range)
cpdef np.ndarray sample2d_points(object function2d, object points)
cpdef np.ndarray sample2d_grid(object function2d, object x, object y)

cpdef tuple sample3d(object function3d, tuple x_range, tuple y_range, tuple z_range)
cpdef np.ndarray sample3d_points(object function3d, object points)
cpdef np.ndarray sample3d_grid(object function3d, object x, object y, object z)

cpdef tuple samplevector2d(object function2d, tuple x_range, tuple y_range)
cpdef np.ndarray samplevector2d_points(object function2d, object points)
cpdef np.ndarray samplevector2d_grid(object function2d, object x, object y)

cpdef tuple samplevector3d(object function3d, tuple x_range, tuple y_range, tuple z_range)
cpdef np.ndarray samplevector3d_points(object function3d, object points)
cpdef np.ndarray samplevector3d_grid(object function3d, object x, object y, object z)
