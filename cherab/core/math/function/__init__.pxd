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

from raysect.core.math.function.float cimport Function1D, autowrap_function1d
from raysect.core.math.function.float cimport Function2D, autowrap_function2d
from raysect.core.math.function.float cimport Function3D, autowrap_function3d
from raysect.core.math.function.float cimport Constant1D, Constant2D, Constant3D
from raysect.core.math.function.float cimport Discrete2DMesh, Interpolator2DMesh
from raysect.core.math.function.vector3d cimport Function2D as VectorFunction2D
from raysect.core.math.function.vector3d cimport Constant2D as ConstantVector2D
from raysect.core.math.function.vector3d cimport autowrap_function2d as autowrap_vectorfunction2d
from raysect.core.math.function.vector3d cimport FloatToVector3DFunction2D
from raysect.core.math.function.vector3d cimport Function3D as VectorFunction3D
from raysect.core.math.function.vector3d cimport Constant3D as ConstantVector3D
from raysect.core.math.function.vector3d cimport autowrap_function3d as autowrap_vectorfunction3d
from raysect.core.math.function.vector3d cimport FloatToVector3DFunction3D

# Alias deprecated names for FloatToVector3DFunctionxD
ctypedef FloatToVector3DFunction2D ScalarToVectorFunction2D
ctypedef FloatToVector3DFunction3D ScalarToVectorFunction3D
