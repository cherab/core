
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

from raysect.core.ray import Ray as CoreRay
from raysect.optical.material.material import NullMaterial


def find_wall_intersection(world, centre_point, sightline_vec, delta=1E-3):

    while True:

        # Find the next intersection point of the ray with the world
        intersection = world.hit(CoreRay(centre_point, sightline_vec))

        if intersection is None:
            raise ValueError('No intersection with solid material found.')

        elif isinstance(intersection.primitive.material, NullMaterial):
            centre_point += sightline_vec * delta
            continue

        else:
            hit_point = intersection.hit_point.transform(intersection.primitive_to_world)
            return hit_point, intersection.primitive
