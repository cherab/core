# Copyright 2016-2021 Euratom
# Copyright 2016-2021 United Kingdom Atomic Energy Authority
# Copyright 2016-2021 Centro de Investigaciones Energéticas, Medioambientales y Tecnológicas
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

import warnings

from .targetedpixel import TargetedPixelGroup as _TargetedPixelGroup


class TargettedPixelGroup(_TargetedPixelGroup):
    """
    A group of targeted pixel under a single scene-graph node.

    .. deprecated::
        TargettedPixelGroup is deprecated and will be removed in a future version.
        Use TargetedPixelGroup instead.

    A scene-graph object regrouping a series of 'TargetedPixel'
    observers as a scene-graph parent. Allows combined observation and display
    control simultaneously.

    :ivar list x_width: Width of pixel along local x axis
    :ivar list y_width: Width of pixel along local y axis
    :ivar list targets: Targets for preferential sampling
    :ivar list targetted_path_prob: Probability of ray being casted at the target
    """

    def __init__(self, *args, **kwargs):
        warnings.warn(
            "TargettedPixelGroup is deprecated and will be removed in a future version. Use TargetedPixelGroup instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        super().__init__(*args, **kwargs)

    @property
    def targetted_path_prob(self):
        return self.targeted_path_prob

    @targetted_path_prob.setter
    def targetted_path_prob(self, value):
        self.targeted_path_prob = value
