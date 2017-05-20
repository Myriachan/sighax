# sighax
A program for brute-forcing the signature for the "sighax" exploit of a certain device's boot ROM.

"sighax" is the CPU-only version.  "sighaxGPU" is the CUDA-based one for Nvidia GPUs.

Neither one is multithreaded; run multiple instances if you want to.  sighaxGPU takes a "--gpu=" parameter to select which GPU to use for that instance.

Confirmed to work for Windows (Visual C++ 2015 + CUDA SDK 8.0) and for Linux with CUDA installed.
