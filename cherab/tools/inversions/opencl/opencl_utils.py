# -*- coding: utf-8 -*-
# Created by Vladislav Neverov (NRC "Kurchatov Institute") for CHERAB Spectroscopy Modelling Framework
from __future__ import print_function
import warnings
import pyopencl as cl


def get_flops(device, verbose=True):
    """
    Returns the theoretical peak performance of specified OpenCL-compatible GPU or accelerator.
    Currently supports only Nvidia, AMD, Intel or Mali GPUs.

    :param pyopencl.Device device: OpenCL device.
    :param bool verbose: Verbose output, defaults to `verbose=True`.

    :return: Theoretical peak performance in GFLOPs.
    """
    comp_units = device.get_info(cl.device_info.MAX_COMPUTE_UNITS)
    gpu_clock = device.get_info(cl.device_info.MAX_CLOCK_FREQUENCY)
    vendor = device.get_info(cl.device_info.VENDOR).lower()
    gflops = 0
    if "nvidia" in vendor:
        cc_maj = device.get_info(cl.device_info.COMPUTE_CAPABILITY_MAJOR_NV)
        cc_min = device.get_info(cl.device_info.COMPUTE_CAPABILITY_MINOR_NV)
        alu_lanes = 128
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
        warnings.warn('Unsupported device vendor: %s. Unable to estimate theoretical peak performance.' % vendor)
        return 0
    if verbose:
        print("Number of compute units: %d" % comp_units)
        print("GPU maximum clock rate: %d MHz" % gpu_clock)
        print("Estimated theoretical peak performance: %g GFLOPS" % gflops)

    return gflops


def get_best_gpu(platforms=None, device_type=cl.device_type.GPU | cl.device_type.ACCELERATOR, verbose=True):
    """
    Finds the fastest (in terms of theoretical peak performance) GPU and/or accelerator available in specified OpenCL platforms

    :param list platforms: List of pyopencl.Platform instances. Default value: `platforms=None` (all available OpenCL platfroms).
    :param pyopencl.device_type device_type: OpenCL device type (GPU, ACCELERATOR, or both).
        Default value: `device_type=pyopencl.device_type.GPU | pyopencl.device_type.ACCELERATOR`.
    :param bool verbose: Verbose output, defaults to `verbose=True`.

    :return: The pyopencl.Device instance corresponding to the fastest GPU or accelerator available in the specified OpenCL platforms.
    """
    if device_type == cl.device_type.DEFAULT:
        device_type = cl.device_type.GPU | cl.device_type.ACCELERATOR
    if not (cl.device_type.GPU | cl.device_type.ACCELERATOR) & device_type:
        raise ValueError('This function works with GPU devices only')
    if platforms is None:
        platforms = cl.get_platforms()
    if verbose:
        print("Selecting best GPU")
    device_best = None
    max_gflops = 0
    for iplat, platform in enumerate(platforms):
        if verbose:
            print("\nOpenCL platform %d: %s" % (iplat, platform.get_info(cl.platform_info.NAME)))
        devices = platform.get_devices(device_type=device_type)
        for idev, device in enumerate(devices):
            if verbose:
                print("\nDevice %d: %s %s" % (idev, device.get_info(cl.device_info.VENDOR), device.get_info(cl.device_info.NAME)))
            gflops = get_flops(device, verbose)
            if gflops > max_gflops:
                device_best = device
    if device_best is None:
        print("No supported GPUs found\n")
        return None
    if verbose:
        print("\nSelected device: %s %s\n" % (device_best.get_info(cl.device_info.VENDOR), device_best.get_info(cl.device_info.NAME)))
    return device_best


def get_first_device(platforms=None, device_type=cl.device_type.GPU | cl.device_type.ACCELERATOR, verbose=True):
    """
    Returns the first OpenCL device of specified type available in specified OpenCL platforms

    :param list platforms: List of pyopencl.Platform instances. Default value: `platforms=None` (all available OpenCL platfroms).
    :param pyopencl.device_type device_type: OpenCL device type (GPU, ACCELERATOR, or both).
        Default value: `device_type=pyopencl.device_type.GPU | pyopencl.device_type.ACCELERATOR`.
    :param bool verbose: Verbose output, defaults to `verbose=True`.

    :return: The pyopencl.Device instance corresponding to the first device available in the specified OpenCL platforms.
    """
    if platforms is None:
        platforms = cl.get_platforms()
    for platform in platforms:
        devices = platform.get_devices(device_type=device_type)
        if len(devices):
            device = devices[0]
            if verbose:
                print("Selected OpenCL device: %s %s\n" % (device.get_info(cl.device_info.VENDOR), device.get_info(cl.device_info.NAME)))
            return device
    print("\nThere are no devices of specified type\n")

    return None


def device_select(platfrom_id=None, device_id=None, device_type=cl.device_type.GPU | cl.device_type.ACCELERATOR, verbose=True):
    """
    OpenCL device selector. Returns the most powerfull OpenCL device availabe if device_type is GPU or accelerator
    or the first OpenCL device available if device_type is CPU.

    :param int platfrom_id: OpenCL platform ID, defaults to `platfrom_id=None`.
    :param int device_id: OpenCL device ID (in the selected OpenCL platform), defaults to `device_id=None`.
    :param pyopencl.device_type device_type: OpenCL device type (GPU, ACCELERATOR, etc.).
        Default value: `device_type=pyopencl.device_type.GPU | pyopencl.device_type.ACCELERATOR`.
    :param bool verbose: Verbose output, defaults to `verbose=True`.

    :return: The pyopencl.Device instance corresponding to the selected OpenCL device.
    """
    platforms = cl.get_platforms()
    n_platforms = len(platforms)
    if device_type == cl.device_type.DEFAULT:
        device_type = cl.device_type.GPU | cl.device_type.ACCELERATOR
    non_gpu_device = not (cl.device_type.GPU | cl.device_type.ACCELERATOR) & device_type
    if platfrom_id is None:
        if non_gpu_device:
            return get_first_device(platforms, device_type, verbose)
        return get_best_gpu(platforms, device_type, verbose)
    if platfrom_id < n_platforms:
        platform = platforms[platfrom_id]
        devices = platform.get_devices(device_type=device_type)
        n_devices = len(devices)
        if device_id is None:
            if non_gpu_device:
                return get_first_device([platform], device_type, verbose)
            return get_best_gpu([platform], device_type, verbose)
        if device_id < n_devices:
            device = devices[device_id]
            if verbose:
                print("Selected OpenCL device: %s %s" % (device.get_info(cl.device_info.VENDOR), device.get_info(cl.device_info.NAME)))
            if non_gpu_device:
                return (device, 0)
            return (device, get_flops(device, verbose))
        warnings.warn('%s platform has %d devices of specified type\n' % (platform.get_info(cl.platform_info.NAME), n_platforms))
        if non_gpu_device:
            return get_first_device([platform], device_type, verbose)
        return get_best_gpu([platform], device_type, verbose)
    warnings.warn('System has only %d OpenCL platforms\n' % n_platforms)
    if non_gpu_device:
        return get_first_device(platforms, device_type, verbose)

    return get_best_gpu(platforms, device_type, verbose)


if __name__ == "__main__":
    device_select(platfrom_id=None, device_id=None, verbose=True)
