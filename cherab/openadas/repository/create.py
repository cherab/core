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

from cherab.core.utility import RecursiveDict
from cherab.core.atomic.elements import *
from cherab.openadas.install import install_files
from cherab.openadas import repository


def populate(download=True, repository_path=None, adas_path=None):
    """
    Populates the OpenADAS repository with a typical set of rates and wavelengths.

    If an ADAS file is not note found an attempt will be made to download the
    file from the OpenADAS website. This behaviour can be disabled by setting
    the download argument to False.

    :param download: Attempt to download the ADAS files if missing (default=True).
    :param repository_path: Alternate path for the OpenADAS repository (default=None).
    :param adas_path: Alternate path in which to search for ADAS files (default=None) .
    """

    # install a common selection of open adas files
    rates = {
        'adf11scd': (
            (hydrogen, 'adf11/scd12/scd12_h.dat'),
            (helium, 'adf11/scd96/scd96_he.dat'),
            (lithium, 'adf11/scd96/scd96_li.dat'),
            (beryllium, 'adf11/scd96/scd96_be.dat'),
            (boron, 'adf11/scd89/scd89_b.dat'),
            (carbon, 'adf11/scd96/scd96_c.dat'),
            (nitrogen, 'adf11/scd96/scd96_n.dat'),
            (oxygen, 'adf11/scd96/scd96_o.dat'),
            (neon, 'adf11/scd96/scd96_ne.dat'),
            (argon, 'adf11/scd89/scd89_ar.dat'),
            (krypton, 'adf11/scd89/scd89_kr.dat'),
            (xenon, 'adf11/scd89/scd89_xe.dat'),
        ),
        'adf11acd': (
            (hydrogen, 'adf11/acd12/acd12_h.dat'),
            (helium, 'adf11/acd96/acd96_he.dat'),
            (lithium, 'adf11/acd96/acd96_li.dat'),
            (beryllium, 'adf11/acd96/acd96_be.dat'),
            (boron, 'adf11/acd89/acd89_b.dat'),
            (carbon, 'adf11/acd96/acd96_c.dat'),
            (nitrogen, 'adf11/acd96/acd96_n.dat'),
            (oxygen, 'adf11/acd96/acd96_o.dat'),
            (neon, 'adf11/acd96/acd96_ne.dat'),
            (argon, 'adf11/acd89/acd89_ar.dat'),
            (krypton, 'adf11/acd89/acd89_kr.dat'),
            (xenon, 'adf11/acd89/acd89_xe.dat'),
        ),
        'adf11ccd': (
            # (donor_element, donor_charge, receiver_element, file_path)
            (hydrogen, 0, hydrogen, 'adf11/ccd96/ccd96_h.dat'),
            (hydrogen, 0, helium, 'adf11/ccd96/ccd96_he.dat'),
            (hydrogen, 0, lithium, 'adf11/ccd89/ccd89_li.dat'),
            (hydrogen, 0, beryllium, 'adf11/ccd89/ccd89_be.dat'),
            (hydrogen, 0, boron, 'adf11/ccd89/ccd89_b.dat'),
            (hydrogen, 0, carbon, 'adf11/ccd96/ccd96_c.dat'),
            (hydrogen, 0, nitrogen, 'adf11/ccd89/ccd89_n.dat'),
            (hydrogen, 0, oxygen, 'adf11/ccd89/ccd89_o.dat'),
            (hydrogen, 0, neon, 'adf11/ccd89/ccd89_ne.dat'),
            (hydrogen, 0, argon, 'adf11/ccd89/ccd89_ar.dat'),
            (hydrogen, 0, krypton, 'adf11/ccd89/ccd89_kr.dat'),
            (hydrogen, 0, xenon, 'adf11/ccd89/ccd89_xe.dat'),
        ),
        'adf11plt': (
            (hydrogen, 'adf11/plt12/plt12_h.dat'),
            (helium, 'adf11/plt96/plt96_he.dat'),
            (lithium, 'adf11/plt96/plt96_li.dat'),
            (beryllium, 'adf11/plt96/plt96_be.dat'),
            (boron, 'adf11/plt89/plt89_b.dat'),
            (carbon, 'adf11/plt96/plt96_c.dat'),
            (nitrogen, 'adf11/plt96/plt96_n.dat'),
            (oxygen,  'adf11/plt96/plt96_o.dat'),
            (neon, 'adf11/plt96/plt96_ne.dat'),
            (argon, 'adf11/plt40/plt40_ar.dat'),
            (krypton, 'adf11/plt89/plt89_kr.dat'),
            (xenon, 'adf11/plt89/plt89_xe.dat')
        ),
        'adf11prb': (
            (hydrogen, 'adf11/prb12/prb12_h.dat'),
            (helium, 'adf11/prb96/prb96_he.dat'),
            (lithium, 'adf11/prb96/prb96_li.dat'),
            (beryllium, 'adf11/prb96/prb96_be.dat'),
            (boron, 'adf11/prb89/prb89_b.dat'),
            (carbon, 'adf11/prb96/prb96_c.dat'),
            (nitrogen, 'adf11/prb96/prb96_n.dat'),
            (oxygen, 'adf11/prb96/prb96_o.dat'),
            (neon, 'adf11/prb96/prb96_ne.dat'),
            (argon, 'adf11/prb89/prb89_ar.dat'),
            (krypton, 'adf11/prb89/prb89_kr.dat'),
            (xenon, 'adf11/prb89/prb89_xe.dat')
        ),
        'adf11prc': (
            (hydrogen, 'adf11/prc96/prc96_h.dat'),
            (helium, 'adf11/prc96/prc96_he.dat'),
            (lithium, 'adf11/prc89/prc89_li.dat'),
            (beryllium, 'adf11/prc89/prc89_be.dat'),
            (boron, 'adf11/prc89/prc89_b.dat'),
            (carbon, 'adf11/prc96/prc96_c.dat'),
            (nitrogen, 'adf11/prc89/prc89_n.dat'),
            (oxygen, 'adf11/prc89/prc89_o.dat'),
            (neon, 'adf11/prc89/prc89_ne.dat'),
            (argon, 'adf11/prc89/prc89_ar.dat'),
            (krypton, 'adf11/prc89/prc89_kr.dat'),
            (xenon, 'adf11/prc89/prc89_xe.dat')
        ),
        'adf12': (
            # (donor, receiver, ionisation, donor_metastable, rate file)
            (hydrogen, 1, hydrogen,  1, 'adf12/qef93#h/qef93#h_h1.dat'),
            (hydrogen, 1, helium,    2, "adf12/qef93#h/qef93#h_he2.dat"),
            (hydrogen, 2, helium,    2, "adf12/qef97#h/qef97#h_en2_kvi#he2.dat"),
            (hydrogen, 1, beryllium, 4, "adf12/qef93#h/qef93#h_be4.dat"),
            (hydrogen, 2, beryllium, 4, "adf12/qef97#h/qef97#h_en2_kvi#be4.dat"),
            (hydrogen, 1, boron,     5, "adf12/qef93#h/qef93#h_b5.dat"),
            (hydrogen, 2, boron,     5, "adf12/qef97#h/qef97#h_en2_kvi#b5.dat"),
            (hydrogen, 1, carbon,    6, "adf12/qef93#h/qef93#h_c6.dat"),
            (hydrogen, 2, carbon,    6, "adf12/qef97#h/qef97#h_en2_kvi#c6.dat"),
            (hydrogen, 1, neon,      10, "adf12/qef93#h/qef93#h_ne10.dat"),
            (hydrogen, 2, neon,      10, "adf12/qef97#h/qef97#h_en2_kvi#ne10.dat")
        ),
        'adf15': (
            (hydrogen,  0, 'adf15/pec12#h/pec12#h_pju#h0.dat'),
            (helium,    0, 'adf15/pec96#he/pec96#he_pju#he0.dat'),
            (helium,    1, 'adf15/pec96#he/pec96#he_pju#he1.dat'),
            (beryllium, 0, 'adf15/pec96#be/pec96#be_pju#be0.dat'),
            (beryllium, 1, 'adf15/pec96#be/pec96#be_pju#be1.dat'),
            (beryllium, 2, 'adf15/pec96#be/pec96#be_pju#be2.dat'),
            (beryllium, 3, 'adf15/pec96#be/pec96#be_pju#be3.dat'),
            (carbon,    0, 'adf15/pec96#c/pec96#c_vsu#c0.dat'),
            (carbon,    1, 'adf15/pec96#c/pec96#c_vsu#c1.dat'),
            (carbon,    2, 'adf15/pec96#c/pec96#c_vsu#c2.dat'),
            # (neon,      0, 'adf15/pec96#ne/pec96#ne_pju#ne0.dat'),     #TODO: OPENADAS DATA CORRUPT
            # (neon,      1, 'adf15/pec96#ne/pec96#ne_pju#ne1.dat'),     #TODO: OPENADAS DATA CORRUPT
            (nitrogen,  0, 'adf15/pec96#n/pec96#n_vsu#n0.dat'),
            (nitrogen,  1, 'adf15/pec96#n/pec96#n_vsu#n1.dat'),
            # (nitrogen,  2, 'adf15/pec96#n/pec96#n_vsu#n2.dat'),    #TODO: OPENADAS DATA CORRUPT
        ),
        'adf21': (
            # (beam_species, target_ion, target_ionisation, rate file)
            (hydrogen, hydrogen,  1,  "adf21/bms97#h/bms97#h_h1.dat"),
            (hydrogen, helium,    2,  "adf21/bms97#h/bms97#h_he2.dat"),
            (hydrogen, lithium,   3,  "adf21/bms97#h/bms97#h_li3.dat"),
            (hydrogen, beryllium, 4,  "adf21/bms97#h/bms97#h_be4.dat"),
            (hydrogen, boron,     5,  "adf21/bms97#h/bms97#h_b5.dat"),
            (hydrogen, carbon,    6,  "adf21/bms97#h/bms97#h_c6.dat"),
            (hydrogen, nitrogen,  7,  "adf21/bms97#h/bms97#h_n7.dat"),
            (hydrogen, oxygen,    8,  "adf21/bms97#h/bms97#h_o8.dat"),
            (hydrogen, fluorine,  9,  "adf21/bms97#h/bms97#h_f9.dat"),
            (hydrogen, neon,      10, "adf21/bms97#h/bms97#h_ne10.dat"),
        ),
        'adf22bmp': (
            # (beam species, beam metastable, target ion, target ionisation, rate file)
            (hydrogen, 2, hydrogen,  1,  "adf22/bmp97#h/bmp97#h_2_h1.dat"),
            (hydrogen, 3, hydrogen,  1,  "adf22/bmp97#h/bmp97#h_3_h1.dat"),
            (hydrogen, 4, hydrogen,  1,  "adf22/bmp97#h/bmp97#h_4_h1.dat"),
            (hydrogen, 2, helium,    2,  "adf22/bmp97#h/bmp97#h_2_he2.dat"),
            (hydrogen, 3, helium,    2,  "adf22/bmp97#h/bmp97#h_3_he2.dat"),
            (hydrogen, 4, helium,    2,  "adf22/bmp97#h/bmp97#h_4_he2.dat"),
            (hydrogen, 2, lithium,   3,  "adf22/bmp97#h/bmp97#h_2_li3.dat"),
            (hydrogen, 3, lithium,   3,  "adf22/bmp97#h/bmp97#h_3_li3.dat"),
            (hydrogen, 4, lithium,   3,  "adf22/bmp97#h/bmp97#h_4_li3.dat"),
            (hydrogen, 2, beryllium, 4,  "adf22/bmp97#h/bmp97#h_2_be4.dat"),
            (hydrogen, 3, beryllium, 4,  "adf22/bmp97#h/bmp97#h_3_be4.dat"),
            (hydrogen, 4, beryllium, 4,  "adf22/bmp97#h/bmp97#h_4_be4.dat"),
            (hydrogen, 2, boron,     5,  "adf22/bmp97#h/bmp97#h_2_b5.dat"),
            (hydrogen, 3, boron,     5,  "adf22/bmp97#h/bmp97#h_3_b5.dat"),
            (hydrogen, 4, boron,     5,  "adf22/bmp97#h/bmp97#h_4_b5.dat"),
            (hydrogen, 2, carbon,    6,  "adf22/bmp97#h/bmp97#h_2_c6.dat"),
            (hydrogen, 3, carbon,    6,  "adf22/bmp97#h/bmp97#h_3_c6.dat"),
            (hydrogen, 4, carbon,    6,  "adf22/bmp97#h/bmp97#h_4_c6.dat"),
            (hydrogen, 2, nitrogen,  7,  "adf22/bmp97#h/bmp97#h_2_n7.dat"),
            (hydrogen, 3, nitrogen,  7,  "adf22/bmp97#h/bmp97#h_3_n7.dat"),
            (hydrogen, 4, nitrogen,  7,  "adf22/bmp97#h/bmp97#h_4_n7.dat"),
            (hydrogen, 2, oxygen,    8,  "adf22/bmp97#h/bmp97#h_2_o8.dat"),
            (hydrogen, 3, oxygen,    8,  "adf22/bmp97#h/bmp97#h_3_o8.dat"),
            (hydrogen, 4, oxygen,    8,  "adf22/bmp97#h/bmp97#h_4_o8.dat"),
            (hydrogen, 2, fluorine,  9,  "adf22/bmp97#h/bmp97#h_2_f9.dat"),
            (hydrogen, 3, fluorine,  9,  "adf22/bmp97#h/bmp97#h_3_f9.dat"),
            (hydrogen, 4, fluorine,  9,  "adf22/bmp97#h/bmp97#h_4_f9.dat"),
            (hydrogen, 2, neon,      10, "adf22/bmp97#h/bmp97#h_2_ne10.dat"),
            (hydrogen, 3, neon,      10, "adf22/bmp97#h/bmp97#h_3_ne10.dat"),
            (hydrogen, 4, neon,      10, "adf22/bmp97#h/bmp97#h_4_ne10.dat"),
        ),
        'adf22bme': (
            # (beam species, target_ion, target_ionisation, (initial_level, final_level), rate file)
            (hydrogen, hydrogen,  1,  (3, 2), "adf22/bme10#h/bme10#h_h1.dat"),
            (hydrogen, helium,    2,  (3, 2), "adf22/bme97#h/bme97#h_he2.dat"),
            (hydrogen, lithium,   3,  (3, 2), "adf22/bme97#h/bme97#h_li3.dat"),
            (hydrogen, beryllium, 4,  (3, 2), "adf22/bme97#h/bme97#h_be4.dat"),
            (hydrogen, boron,     5,  (3, 2), "adf22/bme97#h/bme97#h_b5.dat"),
            (hydrogen, carbon,    6,  (3, 2), "adf22/bme97#h/bme97#h_c6.dat"),
            (hydrogen, nitrogen,  7,  (3, 2), "adf22/bme97#h/bme97#h_n7.dat"),
            (hydrogen, oxygen,    8,  (3, 2), "adf22/bme97#h/bme97#h_o8.dat"),
            (hydrogen, fluorine,  9,  (3, 2), "adf22/bme97#h/bme97#h_f9.dat"),
            (hydrogen, neon,      10, (3, 2), "adf22/bme97#h/bme97#h_ne10.dat"),
            (hydrogen, argon,     18, (3, 2), "adf22/bme99#h/bme99#h_ar18.dat"),
        )
    }

    # add common wavelengths to the repository
    wavelengths = RecursiveDict()

    # H0
    wavelengths[hydrogen][0] = {
        (2, 1): 121.52,
        (3, 1): 102.53,
        (3, 2): 656.19,
        (4, 1): 97.21,
        (4, 2): 486.06,
        (4, 3): 1874.82,
        (5, 1): 94.93,
        (5, 2): 433.99,
        (5, 3): 1281.61,
        (5, 4): 4050.53,
        (6, 1): 93.74,
        (6, 2): 410.12,
        (6, 3): 1093.64,
        (6, 4): 2624.75,
        (6, 5): 7456.66,
        (7, 1): 93.04,
        (7, 2): 396.95,
        (7, 3): 1004.79,
        (7, 4): 2165.19,
        (7, 5): 4651.78,
        (7, 6): 12366.59,
        (8, 1): 92.58,
        (8, 2): 388.85,
        (8, 3): 954.45,
        (8, 4): 1944.26,
        (8, 5): 3738.95,
        (8, 6): 7499.27,
        (8, 7): 19053.71,
        (9, 1): 92.28,
        (9, 2): 383.49,
        (9, 3): 922.76,
        (9, 4): 1817.13,
        (9, 5): 3295.58,
        (9, 6): 5905.68,
        (9, 7): 11303.84,
        (9, 8): 27791.42,
        (10, 1): 92.06,
        (10, 2): 379.74,
        (10, 3): 901.35,
        (10, 4): 1735.94,
        (10, 5): 3037.9,
        (10, 6): 5126.46,
        (10, 7): 8756.3,
        (10, 8): 16202.13,
        (10, 9): 38853.14,
        (11, 1): 91.9,
        (11, 2): 377.01,
        (11, 3): 886.14,
        (11, 4): 1680.39,
        (11, 5): 2871.76,
        (11, 6): 4670.5,
        (11, 7): 7504.88,
        (11, 8): 12381.84,
        (11, 9): 22330.84,
        (11, 10): 52512.27,
        (12, 1): 91.77,
        (12, 2): 374.96,
        (12, 3): 874.92,
        (12, 4): 1640.47,
        (12, 5): 2757.09,
        (12, 6): 4374.58,
        (12, 7): 6769.08,
        (12, 8): 10498.98,
        (12, 9): 16873.36,
        (12, 10): 29826.65,
        (12, 11): 69042.22,
    }

    # todo: temporary - add more accurate values
    wavelengths[deuterium][0] = wavelengths[hydrogen][0]
    wavelengths[tritium][0] = wavelengths[hydrogen][0]

    # He1+
    wavelengths[helium][1] = {
        (2, 1): 30.378,     # 2p -> 1s
        (3, 1): 25.632,     # 3p -> 1s
        (3, 2): 164.04,     # 3d -> 2p
        (4, 2): 121.51,     # 4d -> 2p
        (4, 3): 468.71,     # 4f -> 3d
        (5, 3): 320.28,     # 5f -> 3d
        (5, 4): 1012.65,    # 5g -> 4f
        (6, 4): 656.20,     # 6g -> 4f
        (6, 5): 1864.20,    # 6h -> 5g
        (7, 5): 1162.53,    # from ADAS comment, unknown source
        (7, 6): 3090.55     # from ADAS comment, unknown source
    }

    # Be3+
    wavelengths[beryllium][3] = {
        (3, 1): 6.4065,     # 3p -> 1s
        (3, 2): 41.002,     # 3d -> 2p
        (4, 2): 30.373,     # 4d -> 2p
        (4, 3): 117.16,     # 4f -> 3d
        (5, 3): 80.092,     # 5f -> 3d
        (5, 4): 253.14,     # 5g -> 4f
        (6, 4): 164.03,     # 6g -> 4f
        (6, 5): 466.01,     # 6h -> 5g
        (7, 5): 290.62,     # from ADAS comment, unknown source
        (7, 6): 772.62,     # from ADAS comment, unknown source
        (8, 6): 468.53,     # from ADAS comment, unknown source
        (8, 7): 1190.42     # from ADAS comment, unknown source
    }

    # B4+
    wavelengths[boron][4] = {
        (3, 1): 4.0996,     # 3p -> 1s
        (3, 2): 26.238,     # 3d -> 2p
        (4, 2): 19.437,     # 4d -> 2p
        (4, 3): 74.980,     # 4f -> 3d
        (5, 3): 51.257,     # 5f -> 3d
        (5, 4): 162.00,     # 5g -> 4f
        (6, 4): 104.98,     # 6g -> 4f
        (6, 5): 298.24,     # 6h -> 5g
        (7, 5): 186.05,     # 7h -> 5g
        (7, 6): 494.48,     # 7i -> 6h
        (8, 6): 299.86,     # 8i -> 6h
        (8, 7): 761.87,     # 8k -> 7i
        (9, 7): 451.99,     # 9k -> 7i
        (9, 8): 1111.25     # from ADAS comment, unknown source
    }

    # C5+
    wavelengths[carbon][5] = {
        (4, 2): 13.496,     # 4d -> 2p
        (4, 3): 52.067,     # 4f -> 3d
        (5, 3): 35.594,     # 5f -> 3d
        (5, 4): 112.50,     # 5g -> 4f
        (6, 4): 72.900,     # 6g -> 4f
        (6, 5): 207.11,     # 6h -> 5g
        (7, 5): 129.20,     # from ADAS comment, unknown source
        (7, 6): 343.38,     # from ADAS comment, unknown source
        (8, 6): 208.23,     # from ADAS comment, unknown source
        (8, 7): 529.07,     # from ADAS comment, unknown source
        (9, 7): 313.87,     # from ADAS comment, unknown source
        (9, 8): 771.69,     # from ADAS comment, unknown source
        (10, 8): 449.89,    # from ADAS comment, unknown source
        (10, 9): 1078.86    # from ADAS comment, unknown source
    }

    # Ne9+
    wavelengths[neon][9] = {
        (6, 5): 74.54,      # from ADAS comment, unknown source
        (7, 6): 123.64,     # from ADAS comment, unknown source
        (8, 7): 190.50,     # from ADAS comment, unknown source
        (9, 8): 277.79,     # from ADAS comment, unknown source
        (10, 9): 388.37,    # from ADAS comment, unknown source
        (11, 10): 524.92,   # from ADAS comment, unknown source
        (12, 11): 690.16,   # from ADAS comment, unknown source
        (13, 12): 886.83,   # from ADAS comment, unknown source
        (6, 4): 26.24,      # from ADAS comment, unknown source
        (7, 5): 46.51,      # from ADAS comment, unknown source
        (8, 6): 74.98,      # from ADAS comment, unknown source
        (9, 7): 113.02,     # from ADAS comment, unknown source
        (10, 8): 162.00,    # from ADAS comment, unknown source
        (11, 9): 223.22,    # from ADAS comment, unknown source
        (12, 10): 298.15,   # from ADAS comment, unknown source
        (13, 11): 388.12    # from ADAS comment, unknown source
    }

    install_files(rates, download=download, repository_path=repository_path, adas_path=adas_path)
    repository.update_wavelengths(wavelengths, repository_path=repository_path)
