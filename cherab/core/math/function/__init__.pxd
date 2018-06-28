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

from raysect.core.math.function cimport Function1D, autowrap_function1d
from raysect.core.math.function cimport Function2D, autowrap_function2d
from raysect.core.math.function cimport Function3D, autowrap_function3d
from cherab.core.math.function.vectorfunction2d cimport VectorFunction2D, autowrap_vectorfunction2d, ScalarToVectorFunction2D
from cherab.core.math.function.vectorfunction3d cimport VectorFunction3D, autowrap_vectorfunction3d, ScalarToVectorFunction3D