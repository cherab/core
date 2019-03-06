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

# todo: this code is now incomplete it should return an equilibrium object
# import re
# import numpy as np
# from cherab.core.math import Interpolate2DCubic
#
#
# def import_equ_psi(equ_file):
#     """
#     Imports a 2D Psi grid from an equ mesh equilibrium.
#
#     :param str equ_file: The .EQU mesh generator equilibrium file.
#     :return: A 2D Psi(r,z) function
#     :rtype: Interpolate2DCubic
#     """
#
#     fh = open(equ_file, 'r')
#     file_lines = fh.readlines()
#     fh.close()
#
#     # Load r array
#     line_i = 0
#     while True:
#         line = file_lines[line_i]
#         match = re.match('^\s*r\(1:jm\);', line)
#
#         if not match:
#             line_i += 1
#             continue
#
#         line_i += 1
#         r_values = []
#         line = file_lines[line_i]
#         while line and not line.isspace():
#             print(line)
#             values = line.split()
#             print(values)
#             for v in values:
#                 r_values.append(float(v))
#
#             line_i += 1
#             line = file_lines[line_i]
#
#         jm = len(r_values)
#         break
#     r_values = np.array(r_values)
#
#     # Load z array
#     while True:
#         line = file_lines[line_i]
#         match = re.match('^\s*z\(1:km\);', line)
#
#         if not match:
#             line_i += 1
#             continue
#
#         line_i += 1
#         z_values = []
#         line = file_lines[line_i]
#         while not re.match('^\s*$', line):
#             values = line.split()
#             for v in values:
#                 z_values.append(float(v))
#
#             line_i += 1
#             line = file_lines[line_i]
#
#         km = len(z_values)
#         break
#     z_values = np.array(z_values)
#
#     # Load (r, z) array
#     while True:
#         line = file_lines[line_i]
#         match = re.match('^\s*\(\(psi\(j,k\)-psib,j=1,jm\),k=1,km\)', line)
#
#         if not match:
#             line_i += 1
#             continue
#
#         line_i += 1
#         psi_values = []
#         line = file_lines[line_i]
#         while not re.match('^\s*$', line):
#             values = line.split()
#             for v in values:
#                 psi_values.append(float(v))
#
#             line_i += 1
#             try:
#                 line = file_lines[line_i]
#             except IndexError:
#                 break
#
#         if len(psi_values) != km * jm:
#             raise ValueError("Number of values in r, z array not equal to (km, jm).")
#
#         break
#
#     psi_raw = np.array(psi_values)
#     psi = psi_raw.reshape((km, jm))
#     psi = np.swapaxes(psi, 0, 1)
#
#     return Interpolate2DCubic(r_values, z_values, psi)
