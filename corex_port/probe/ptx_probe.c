/*
 * ptx_probe.c — Empirical probe of the ivcore11 (Iluvatar CoreX) driver's
 * handling of NVIDIA PTX text via cuModuleLoadData / cuModuleLoadDataEx.
 *
 * This is the exact code path CUDA.jl relies on: it compiles Julia -> LLVM
 * NVPTX -> PTX text and hands that PTX to the driver via cuModuleLoadData(Ex).
 * ILGPU / TornadoVM previously observed CUDA_ERROR_INVALID_IMAGE (200) here.
 *
 * Build:  clang ptx_probe.c -I$COREX_PATH/include -L$COREX_PATH/lib -lcuda -o ptx_probe
 * Run:    ./ptx_probe
 */
#include <stdio.h>
#include <string.h>
#include <cuda.h>

/* Minimal, valid NVIDIA PTX: an empty kernel. Accepted by any real NV driver. */
static const char *PTX =
    ".version 6.0\n"
    ".target sm_52\n"
    ".address_size 64\n"
    "\n"
    ".visible .entry probe_kernel()\n"
    "{\n"
    "    ret;\n"
    "}\n";

static void report(const char *what, CUresult rc) {
    const char *name = NULL, *str = NULL;
    cuGetErrorName(rc, &name);
    cuGetErrorString(rc, &str);
    printf("%-24s rc=%d name=%s msg=%s\n", what, (int)rc,
           name ? name : "?", str ? str : "?");
}

int main(void) {
    CUresult rc;
    int drv = 0;

    rc = cuInit(0);
    report("cuInit", rc);
    if (rc != CUDA_SUCCESS) return 1;

    rc = cuDriverGetVersion(&drv);
    report("cuDriverGetVersion", rc);
    printf("driver_version_int=%d (i.e. CUDA %d.%d)\n", drv, drv/1000, (drv%1000)/10);

    int ndev = 0;
    rc = cuDeviceGetCount(&ndev);
    report("cuDeviceGetCount", rc);
    printf("device_count=%d\n", ndev);
    if (ndev < 1) return 1;

    CUdevice dev;
    rc = cuDeviceGet(&dev, 0);
    report("cuDeviceGet", rc);

    char devname[256] = {0};
    cuDeviceGetName(devname, sizeof(devname), dev);
    printf("device0_name=%s\n", devname);

    CUcontext ctx;
    rc = cuCtxCreate(&ctx, 0, dev);
    report("cuCtxCreate", rc);
    if (rc != CUDA_SUCCESS) return 1;

    /* THE KEY TEST: load NV PTX text through the driver JIT. */
    CUmodule mod;
    rc = cuModuleLoadData(&mod, PTX);
    report("cuModuleLoadData(PTX)", rc);
    if (rc == CUDA_SUCCESS) {
        printf("RESULT: PTX JIT ACCEPTED — CUDA.jl PTX path may be viable.\n");
        CUfunction fn;
        CUresult r2 = cuModuleGetFunction(&fn, mod, "probe_kernel");
        report("cuModuleGetFunction", r2);
        cuModuleUnload(mod);
    } else {
        printf("RESULT: PTX JIT REJECTED (rc=%d) — driver does not consume NV PTX text.\n", (int)rc);
    }

    /* Also exercise cuModuleLoadDataEx (used by ptxas-style JIT options). */
    CUmodule mod2;
    rc = cuModuleLoadDataEx(&mod2, PTX, 0, NULL, NULL);
    report("cuModuleLoadDataEx(PTX)", rc);
    if (rc == CUDA_SUCCESS) cuModuleUnload(mod2);

    cuCtxDestroy(ctx);
    return 0;
}
