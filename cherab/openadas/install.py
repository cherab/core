
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


import os
import urllib.parse
import urllib.request

from cherab.openadas import repository
from cherab.openadas.parse import *
from cherab.core.utility import RecursiveDict, PerCm3ToPerM3, Cm3ToM3


ADAS_DOWNLOAD_CACHE = os.path.expanduser('~/.cherab/openadas/download_cache')
OPENADAS_FILE_URL = 'http://open.adas.ac.uk/download/'


def install_files(configuration, download=False, repository_path=None, adas_path=None):

    for adf in configuration:
        if adf.lower() == 'adf11scd':
            for args in configuration[adf]:
                install_adf11scd(*args, download=download, repository_path=repository_path, adas_path=adas_path)
        if adf.lower() == 'adf11acd':
            for args in configuration[adf]:
                install_adf11acd(*args, download=download, repository_path=repository_path, adas_path=adas_path)
        if adf.lower() == 'adf11ccd':
            for args in configuration[adf]:
                install_adf11ccd(*args, download=download, repository_path=repository_path, adas_path=adas_path)
        if adf.lower() == 'adf11plt':
            for args in configuration[adf]:
                install_adf11plt(*args, download=download, repository_path=repository_path, adas_path=adas_path)
        if adf.lower() == 'adf11prb':
            for args in configuration[adf]:
                install_adf11prb(*args, download=download, repository_path=repository_path, adas_path=adas_path)
        if adf.lower() == 'adf11prc':
            for args in configuration[adf]:
                install_adf11prc(*args, download=download, repository_path=repository_path, adas_path=adas_path)
        if adf.lower() == 'adf12':
            for args in configuration[adf]:
                install_adf12(*args, download=download, repository_path=repository_path, adas_path=adas_path)
        if adf.lower() == 'adf15':
            for args in configuration[adf]:
                install_adf15(*args, download=download, repository_path=repository_path, adas_path=adas_path)
        if adf.lower() == 'adf21':
            for args in configuration[adf]:
                install_adf21(*args, download=download, repository_path=repository_path, adas_path=adas_path)
        if adf.lower() == 'adf22bmp':
            for args in configuration[adf]:
                install_adf22bmp(*args, download=download, repository_path=repository_path, adas_path=adas_path)
        if adf.lower() == 'adf22bme':
            for args in configuration[adf]:
                install_adf22bme(*args, download=download, repository_path=repository_path, adas_path=adas_path)


# todo: move print calls to logging

def install_adf11scd(element, file_path, download=False, repository_path=None, adas_path=None):
    """
    Adds the ionisation rate defined in an ADF11 file to the repository.

    :param element: The element described by the rate file.
    :param file_path: Path relative to ADAS root.
    :param download: Attempt to download file if not present (Default=True).
    :param repository_path: Path to the repository in which to install the rates (optional).
    :param adas_path: Path to ADAS files repository (optional).
    """

    print('Installing {}...'.format(file_path))
    path = _locate_adas_file(file_path, download, adas_path)
    if not path:
        raise ValueError('Could not locate the specified ADAS file.')

    # decode file and write out rates
    rate_adas = parse_adf11(element, path)

    rate_cherab = _notation_adf11_adas2cherab(rate_adas, "scd")  # convert from adas to cherab notation

    repository.update_ionisation_rates(rate_cherab, repository_path)


def install_adf11acd(element, file_path, download=False, repository_path=None, adas_path=None):
    """
    Adds the recombination rate defined in an ADF11 file to the repository.

    :param element: The element described by the rate file.
    :param file_path: Path relative to ADAS root.
    :param download: Attempt to download file if not present (Default=True).
    :param repository_path: Path to the repository in which to install the rates (optional).
    :param adas_path: Path to ADAS files repository (optional).
    """

    print('Installing {}...'.format(file_path))
    path = _locate_adas_file(file_path, download, adas_path)
    if not path:
        raise ValueError('Could not locate the specified ADAS file.')

    # decode file and write out rates
    rate_adas = parse_adf11(element, path)

    rate_cherab = _notation_adf11_adas2cherab(rate_adas, "acd")  # convert from adas to cherab notation

    repository.update_recombination_rates(rate_cherab, repository_path)


def install_adf11ccd(donor_element, donor_charge, receiver_element, file_path, download=False,
                     repository_path=None, adas_path=None):
    """
    Adds the thermal charge exchange rate defined in an ADF11 file to the repository.

    :param donor_element: Element donating the electron, for the case of ADF11 files it is
      neutral hydrogen.
    :param donor_charge: Charge of the donor atom/ion.
    :param receiver_element: Element receiving the electron.
    :param file_path: Path relative to ADAS root.
    :param download: Attempt to download file if not present (Default=True).
    :param repository_path: Path to the repository in which to install the rates (optional).
    :param adas_path: Path to ADAS files repository (optional).
    """

    print('Installing {}...'.format(file_path))
    path = _locate_adas_file(file_path, download, adas_path)
    if not path:
        raise ValueError('Could not locate the specified ADAS file.')

    # decode file and write out rates
    rate_adas = parse_adf11(receiver_element, path)
    rate_cherab = _notation_adf11_adas2cherab(rate_adas, "ccd")  # convert from adas to cherab notation

    # reshape rate dictionary to match cherab convention
    rate_cherab_ccd = RecursiveDict()
    rate_cherab_ccd[donor_element][donor_charge] = rate_cherab

    repository.update_thermal_cx_rates(rate_cherab_ccd, repository_path)


def install_adf11plt(element, file_path, download=False, repository_path=None, adas_path=None):
    """
    Adds the line radiated power rates defined in an ADF11 file to the repository.

    :param element: The element described by the rate file.
    :param file_path: Path relative to ADAS root.
    :param download: Attempt to download file if not present (Default=True).
    :param repository_path: Path to the repository in which to install the rates (optional).
    :param adas_path: Path to ADAS files repository (optional).
    """

    print('Installing {}...'.format(file_path))
    path = _locate_adas_file(file_path, download, adas_path)
    if not path:
        raise ValueError('Could not locate the specified ADAS file.')

    # decode file and write out rates
    rate_adas = parse_adf11(element, path)

    rate_cherab = _notation_adf11_adas2cherab(rate_adas, "plt")  # convert from adas to cherab notation

    repository.update_line_power_rates(rate_cherab, repository_path)


def install_adf11prb(element, file_path, download=False, repository_path=None, adas_path=None):
    """
    Adds the continuum radiated power rates defined in an ADF11 file to the repository.

    :param element: The element described by the rate file.
    :param file_path: Path relative to ADAS root.
    :param download: Attempt to download file if not present (Default=True).
    :param repository_path: Path to the repository in which to install the rates (optional).
    :param adas_path: Path to ADAS files repository (optional).
    """

    print('Installing {}...'.format(file_path))
    path = _locate_adas_file(file_path, download, adas_path)
    if not path:
        raise ValueError('Could not locate the specified ADAS file.')

    # decode file and write out rates
    rate_adas = parse_adf11(element, path)

    rate_cherab = _notation_adf11_adas2cherab(rate_adas, "prb")  # convert from adas to cherab notation

    repository.update_continuum_power_rates(rate_cherab, repository_path)


def install_adf11prc(element, file_path, download=False, repository_path=None, adas_path=None):
    """
    Adds the CX radiated power rates defined in an ADF11 file to the repository.

    :param element: The element described by the rate file.
    :param file_path: Path relative to ADAS root.
    :param download: Attempt to download file if not present (Default=True).
    :param repository_path: Path to the repository in which to install the rates (optional).
    :param adas_path: Path to ADAS files repository (optional).
    """

    print('Installing {}...'.format(file_path))
    path = _locate_adas_file(file_path, download, adas_path)
    if not path:
        raise ValueError('Could not locate the specified ADAS file.')

    # decode file and write out rates
    rate_adas = parse_adf11(element, path)

    rate_cherab = _notation_adf11_adas2cherab(rate_adas, "prc")  # convert from adas to cherab notation

    repository.update_cx_power_rates(rate_cherab, repository_path)


def install_adf12(donor_ion, donor_metastable, receiver_ion, receiver_charge, file_path, download=False, repository_path=None, adas_path=None):
    """
    Adds the rates in the ADF12 file to the repository.

    :param donor_ion: The donor ion element described by the rate file.
    :param donor_metastable: The donor ion metastable level.
    :param receiver_ion: The receiver ion element described by the rate file.
    :param receiver_charge: The receiver ion ionisation level described by the rate file.
    :param file_path: Path relative to ADAS root.
    :param download: Attempt to download file if not present (Default=True).
    :param repository_path: Path to the repository in which to install the rates (optional).
    :param adas_path: Path to ADAS files repository (optional).
    """

    print('Installing {}...'.format(file_path))
    path = _locate_adas_file(file_path, download, adas_path)
    if not path:
        raise ValueError('Could not locate the specified ADAS file.')

    # decode file and write out rates
    rates = parse_adf12(donor_ion, donor_metastable, receiver_ion, receiver_charge, path)
    repository.update_beam_cx_rates(rates, repository_path)


def install_adf15(element, ionisation, file_path, download=False, repository_path=None, adas_path=None, header_format=None):
    """
    Adds the rates in the ADF15 file to the repository.

    :param element: The element described by the rate file.
    :param ionisation: The ionisation level described by the rate file.
    :param file_path: Path relative to ADAS root.
    :param download: Attempt to download file if not present (Default=True).
    :param repository_path: Path to the repository in which to install the rates (optional).
    :param adas_path: Path to ADAS files repository (optional).
    """

    print('Installing {}...'.format(file_path))
    path = _locate_adas_file(file_path, download, adas_path)
    if not path:
        raise ValueError('Could not locate the specified ADAS file.')

    # decode file and write out rates
    rates, wavelengths = parse_adf15(element, ionisation, path, header_format=header_format)
    repository.update_pec_rates(rates, repository_path)
    repository.update_wavelengths(wavelengths, repository_path)


def install_adf21(beam_species, target_ion, target_charge, file_path, download=False, repository_path=None, adas_path=None):
    # """
    # Adds the rate defined in an ADF21 file to the repository.
    #
    # :param file_path: Path relative to ADAS root.
    # :param download: Attempt to download file if not present (Default=True).
    # :param repository_path: Path to the repository in which to install the rates (optional).
    # :param adas_path: Path to ADAS files repository (optional).
    # """

    print('Installing {}...'.format(file_path))
    path = _locate_adas_file(file_path, download, adas_path)
    if not path:
        raise ValueError('Could not locate the specified ADAS file.')

    # # decode file and write out rates
    rate = parse_adf21(beam_species, target_ion, target_charge, path)
    repository.update_beam_stopping_rates(rate, repository_path)


def install_adf22bmp(beam_species, beam_metastable, target_ion, target_charge, file_path, download=False, repository_path=None, adas_path=None):
    pass
    # """
    # Adds the rate defined in an ADF21 file to the repository.
    #
    # :param file_path: Path relative to ADAS root.
    # :param download: Attempt to download file if not present (Default=True).
    # :param repository_path: Path to the repository in which to install the rates (optional).
    # :param adas_path: Path to ADAS files repository (optional).
    # """

    print('Installing {}...'.format(file_path))
    path = _locate_adas_file(file_path, download, adas_path)
    if not path:
        raise ValueError('Could not locate the specified ADAS file.')

    # # decode file and write out rates
    rate = parse_adf22bmp(beam_species, beam_metastable, target_ion, target_charge, path)
    repository.update_beam_population_rates(rate, repository_path)


def install_adf22bme(beam_species, target_ion, target_charge, transition, file_path, download=False, repository_path=None, adas_path=None):
    pass
    # """
    # Adds the rate defined in an ADF21 file to the repository.
    #
    # :param file_path: Path relative to ADAS root.
    # :param download: Attempt to download file if not present (Default=True).
    # :param repository_path: Path to the repository in which to install the rates (optional).
    # :param adas_path: Path to ADAS files repository (optional).
    # """

    print('Installing {}...'.format(file_path))
    path = _locate_adas_file(file_path, download, adas_path)
    if not path:
        raise ValueError('Could not locate the specified ADAS file.')

    # # decode file and write out rates
    rate = parse_adf22bme(beam_species, target_ion, target_charge, transition, path)
    repository.update_beam_emission_rates(rate, repository_path)


def _locate_adas_file(file_path, download=False, adas_path=None):

    path = None

    # is file in adas path?
    if adas_path:
        target = os.path.join(adas_path, file_path)
        if os.path.isfile(target):
            path = target

    # download file?
    if not path and download:
        target = os.path.join(ADAS_DOWNLOAD_CACHE, file_path)

        # is file in cache? if not download...
        if os.path.isfile(target):
            path = target
        else:

            # create directory structure if missing
            directory = os.path.dirname(target)
            if not os.path.isdir(directory):
                os.makedirs(directory)

            print(" - downloading ADF file '{}' to '{}'".format(file_path, target))

            url = urllib.parse.urljoin(OPENADAS_FILE_URL, file_path.replace('#', '][').lstrip('/'))
            urllib.request.urlretrieve(url, target)
            path = target

    return path


def _notation_adf11_adas2cherab(rate_adas, filetype):
    """
    Converts adas unit, charge and numeric notation to cherab notation

    :param rate_adas: Nested dictionary of shape rate_adas[element][charge][te, ne, rates]
    :param filetype: string denoting adas adf11 file type to decide whether charge conversion is to be applied.
      Will be applied for file types: "scd", "ccd", "plt", "pls"
    :return: nested dictionary with cherab rates and units notation
    """

    # Charge correction will be applied if there is difference between adas and cherab charge notation
    if filetype in ["scd", "plt", "pls"]:
        charge_correction = int(-1)
    else:
        charge_correction = int(0)

    # adas units, charge and number notation to be changed to cherab notation
    rate_cherab = RecursiveDict()
    for i in rate_adas.keys():
        for j in rate_adas[i].keys():
            # convert from adas log10 in [cm**-3] notation to cherab [m**-3] electron density notation
            rate_cherab[i][j + charge_correction]["ne"] = PerCm3ToPerM3.to(10**rate_adas[i][j]["ne"])
            # convert from adas log10 to cherab electron temperature notation
            rate_cherab[i][j + charge_correction]["te"] = 10**rate_adas[i][j]["te"]
            rate_cherab[i][j + charge_correction]["rates"] = Cm3ToM3.to(10**rate_adas[i][j]["rates"])

    return rate_cherab


