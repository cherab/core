# -*- coding: utf-8 -*-
#
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

import warnings
try:
    import pyopencl as cl
except ImportError:
    _has_pyopencl = False
else:
    _has_pyopencl = True


def get_flops(device, verbose=False):
    """
    Returns the theoretical peak performance of specified OpenCL-compatible GPU or ACCELERATOR.
    Currently supports only Nvidia, AMD, Intel or Mali GPUs.

    :param pyopencl.Device device: OpenCL device.
    :param bool verbose: Verbose output, defaults to `verbose=False`.

    :return: Theoretical peak performance in GFLOPs.
    """

    if not _has_pyopencl:
        raise RuntimeError("The pyopencl module is required to run get_flops() function.")

    device_type = device.get_info(cl.device_info.TYPE)
    if not (device_type & (cl.device_type.GPU | cl.device_type.ACCELERATOR)):
        raise ValueError("Unsupported device type: {}.".format(cl.device_type.to_string(device_type)))

    comp_units = device.get_info(cl.device_info.MAX_COMPUTE_UNITS)
    gpu_clock = device.get_info(cl.device_info.MAX_CLOCK_FREQUENCY)
    vendor = device.get_info(cl.device_info.VENDOR).lower()
    gflops = 0

    if "nvidia" in vendor:
        cc_maj = device.get_info(cl.device_info.COMPUTE_CAPABILITY_MAJOR_NV)
        cc_min = device.get_info(cl.device_info.COMPUTE_CAPABILITY_MINOR_NV)
        alu_lanes = 128  # default (as in 8.6)
        if cc_maj == 1:
            alu_lanes = 8
        elif cc_maj == 2:
            alu_lanes = 32 if cc_min == 0 else 48
        elif cc_maj == 3:
            alu_lanes = 192
        elif cc_maj == 5:
            alu_lanes = 128
        elif cc_maj == 6:
            alu_lanes = 64 if cc_min == 0 else 128
        elif cc_maj == 7:
            alu_lanes = 64
        elif cc_maj == 8:
            # For the devices with CUDA CC 8.6, the same ALUs can do both integer and float32 math.
            # Theoreticaly this doubles the peak float32 performance as it double the number of ALU lanes with float32 support.
            alu_lanes = 64 if cc_min == 0 else 128
        gflops = comp_units * alu_lanes * 2 * gpu_clock / 1000.

    elif "amd" in vendor or "advanced" in vendor:
        try:
            ww = device.get_info(cl.device_info.WAVEFRONT_WIDTH_AMD)
        except:
            ww = 64
        gflops = comp_units * ww * 2 * gpu_clock / 1000.

    elif "intel" in vendor:
        gflops = comp_units * 16 * gpu_clock / 1000.

    elif "arm" in vendor:
        gflops = comp_units * 2 * 16 * gpu_clock / 1000.

    else:
        warnings.warn('Unsupported device vendor: {}. Unable to estimate theoretical peak performance.'.format(vendor))
        return 0

    if verbose:
        print("Number of compute units: {}".format(comp_units))
        print("GPU maximum clock rate: {} MHz".format(gpu_clock))
        print("Estimated theoretical peak performance: {} GFLOPS".format(gflops))

    return gflops


def get_best_gpu(platforms=None, device_type=None, verbose=False):
    """
    Finds the fastest (in terms of theoretical peak performance) GPU and/or accelerator
    available in specified OpenCL platforms

    :param list platforms: List of pyopencl.Platform instances. Default value: `platforms=None`
                           (all available OpenCL platfroms).
    :param pyopencl.device_type device_type: OpenCL device type (GPU, ACCELERATOR, or both).
                                             Default value: `device_type=None` (GPU or accelerator).
                                             If device_type is ALL or DEFAULT, all non-GPU/ACCELERATOR
                                             devices will be skipped.
    :param bool verbose: Verbose output, defaults to `verbose=False`.

    :return: The pyopencl.Device instance corresponding to the fastest GPU or accelerator available
             in the specified OpenCL platforms.
    """

    if not _has_pyopencl:
        raise RuntimeError("The pyopencl module is required to run get_best_gpu() function.")

    device_type = device_type or cl.device_type.GPU | cl.device_type.ACCELERATOR
    if device_type == cl.device_type.DEFAULT or device_type == cl.device_type.ALL:
        device_type = cl.device_type.GPU | cl.device_type.ACCELERATOR

    if not ((cl.device_type.GPU | cl.device_type.ACCELERATOR) & device_type):
        raise ValueError('This function works with GPU devices only.')

    if verbose:
        print("Selecting best GPU...")

    if platforms is None:
        platforms = cl.get_platforms()
    device_best = None
    max_gflops = 0
    for iplat, platform in enumerate(platforms):
        if verbose:
            print("\nOpenCL platform {}: {}.".format(iplat, platform.get_info(cl.platform_info.NAME)))
        devices = platform.get_devices(device_type=device_type)
        for idev, device in enumerate(devices):
            if verbose:
                print("\nDevice {}: {} {}.".format(idev, device.get_info(cl.device_info.VENDOR),
                                                   device.get_info(cl.device_info.NAME)))
            gflops = get_flops(device, verbose)
            if gflops > max_gflops:
                device_best = device

    if device_best is None:
        warnings.warn("No supported GPUs found.")
        return None

    if verbose:
        print("\nSelected OpenCL device: {} {}.\n".format(device_best.get_info(cl.device_info.VENDOR),
                                                          device_best.get_info(cl.device_info.NAME)))
    return device_best


def get_first_device(platforms=None, device_type=None, verbose=False):
    """
    Returns the first OpenCL device of specified type available in specified OpenCL platforms

    :param list platforms: List of pyopencl.Platform instances. Default value: `platforms=None` (all available OpenCL platfroms).
    :param pyopencl.device_type device_type: OpenCL device type (GPU, ACCELERATOR, CPU, ALL, etc.).
        Default value: `device_type=None` (GPU or accelerator).
    :param bool verbose: Verbose output, defaults to `verbose=False`.

    :return: The pyopencl.Device instance corresponding to the first device available in the specified OpenCL platforms.
    """

    if not _has_pyopencl:
        raise RuntimeError("The pyopencl module is required to run get_first_device() function.")

    device_type = device_type or cl.device_type.GPU | cl.device_type.ACCELERATOR
    if platforms is None:
        platforms = cl.get_platforms()
    for platform in platforms:
        devices = platform.get_devices(device_type=device_type)
        if len(devices):
            device = devices[0]
            if verbose:
                print("Selected OpenCL device: {} {}.\n".format(device.get_info(cl.device_info.VENDOR),
                                                                device.get_info(cl.device_info.NAME)))
            return device

    warnings.warn("Unable to find OpenCL devices of specified type.")

    return None


def device_select(platfrom_id=None, device_id=None, device_type=None, verbose=False):
    """
    OpenCL device selector. Returns the most powerfull OpenCL device availabe
    if device_type is GPU or ACCELERATOR or the first OpenCL device available
    if device_type is CPU, ALL or CUSTOM.

    :param int platfrom_id: OpenCL platform ID, defaults to `platfrom_id=None`.
    :param int device_id: OpenCL device ID (in the selected OpenCL platform),
                          defaults to `device_id=None`.
    :param pyopencl.device_type device_type: OpenCL device type (GPU, ACCELERATOR, etc.).
                                             Default value: `device_type=None` (GPU | ACCELERATOR).
    :param bool verbose: Verbose output, defaults to `verbose=False`.

    :return: The pyopencl.Device instance corresponding to the selected OpenCL device.
    """

    if not _has_pyopencl:
        raise RuntimeError("The pyopencl module is required to run device_select() function.")

    device_type = device_type or cl.device_type.GPU | cl.device_type.ACCELERATOR
    if device_type == cl.device_type.DEFAULT:
        device_type = cl.device_type.GPU | cl.device_type.ACCELERATOR

    non_gpu_device = (device_type == cl.device_type.ALL) or not ((cl.device_type.GPU | cl.device_type.ACCELERATOR) & device_type)

    platforms = cl.get_platforms()
    n_platforms = len(platforms)

    if platfrom_id is not None and platfrom_id >= n_platforms:
        raise ValueError('The platform_id {} exceeds the number of OpenCL platforms ({}) found in the system.'.format(platfrom_id, n_platforms))

    if platfrom_id is None:
        if non_gpu_device:
            return get_first_device(platforms, device_type, verbose)
        # get_best_gpu() returns None for unsupported devices, if so return the first device
        return get_best_gpu(platforms, device_type, verbose) or get_first_device(platforms, device_type, verbose)

    platform = platforms[platfrom_id]
    devices = platform.get_devices(device_type=device_type)
    n_devices = len(devices)

    if device_id is not None and device_id >= n_devices:
        platform_name = platform.get_info(cl.platform_info.NAME)
        raise ValueError('The device_id {} exceeds the number of devices ({}) of specified type in the {} platform.'.format(device_id, n_devices, platform_name))

    if device_id is None:
        if non_gpu_device:
            return get_first_device([platform], device_type, verbose)
        # get_best_gpu() returns None for unsupported devices, if so return the first device
        return get_best_gpu([platform], device_type, verbose) or get_first_device([platform], device_type, verbose)

    device = devices[device_id]
    if verbose:
        print("Selected OpenCL device: {} {}.".format(device.get_info(cl.device_info.VENDOR), device.get_info(cl.device_info.NAME)))
    return device
