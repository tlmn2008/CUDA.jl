# CUDA.jl → Iluvatar CoreX (ivcore11) 迁移记录

## 来源
- 仓库: JuliaGPU/CUDA.jl (https://github.com/JuliaGPU/CUDA.jl)
- 分支/commit: `main` @ `13d5f6be30a07ea63828680839c73125eca0ad6f` (2026-07-24)
- 版本: CUDA.jl 6.2.1 (workspace: CUDACore/CUDATools + lib/{cublas,cusparse,cusolver,cufft,curand,cudnn,cutensor,cutensornet,custatevec,cupti,nvml})

## CUDA 使用性质
CUDA.jl 是 Julia 生态的 CUDA 编程主接口。它**不是**用 nvcc 编译 `.cu`, 而是:
1. 把 Julia 函数经 GPUCompiler.jl + LLVM **NVPTX 后端** 编译成 **NVIDIA PTX 文本**;
2. 用 `ptxas` 把 PTX 汇编成 cubin(或直接把 PTX 交给驱动 JIT);
3. 通过 CUDA driver API `cuLinkAddData(JIT_INPUT_PTX)` / `cuModuleLoadData` / `cuModuleLoadDataEx` 把 PTX/cubin 加载到设备并 `cuLaunchKernel`。

因此其 CUDA 相关性完全建立在「NVIDIA PTX 文本能被驱动 JIT 消费」这一前提上。代码定位:
`CUDACore/src/compiler/compilation.jl`(rewrite_ptx_header/mcgen, 且 `default_ptx_versions` 要求 `ptxas` 支持所需 PTX)、`CUDACore/lib/cudadrv/module/linker.jl`(add_data! 用 JIT_INPUT_PTX)。

## 环境
- `source /home/init_container.sh`: 配好 GITHUB_TOKEN/GITHUB_ORG=tlmn2008 与 CoreX 环境(COREX_PATH=/usr/local/corex, CC/CXX=clang/clang++)。注意脚本用 `set -euo pipefail` 且末尾 apt-get 因锁失败中止, 故实际改为直接 `. /etc/profile.d/{github-migration,corex}.sh`。
- GPU: 2× Iluvatar BI-V150, `ixsmi` 正常(IX-ML 4.4.0, Driver 4.5.0, CUDA 兼容 10.2)。sandbox 内不可见 GPU, 需以完整权限运行。
- Julia: 官方二进制 1.11.5(juliaup 下载被限速, 改从 NJU 镜像取 tar.gz)。Pkg 走 NJU 镜像加速。禁止 nvcc(实测 `/usr/local/corex/bin/nvcc` 是 bash 脚本 stub), 全程 CoreX clang/clang++。未触碰 /usr/local/corex。

## 适配内容
1. **裸探针实证(按要求, 不只下结论)**
   - `probe/ptx_probe.c`: `cuInit`/`cuDriverGetVersion`(=10020)/`cuDeviceGetCount`(=2)/`cuDeviceGet`/`cuCtxCreate` 全部 `IX_SUCCESS`; 对一段**合法最小 NV PTX**(`.version 6.0 .target sm_52` 空 kernel)调用 `cuModuleLoadData` 与 `cuModuleLoadDataEx` 均返回 **rc=200 IX_ERROR_INVALID_IMAGE**("device kernel image is invalid")。→ 驱动不消费 NV PTX 文本。
   - `probe/native_kernel.cu`: 用 `clang++ -x cuda --cuda-gpu-arch=ivcore11` 编译等价 `add1` kernel 并 `<<<>>>` 运行, 结果正确(h[0]=1.0, h[255]=256.0)。→ GPU 与 CoreX 工具链本身正常, 壁垒**专属**于 NV PTX 文本 ingestion(即 CUDA.jl 的必经路径)。
   - 探查兜底: CUDA.jl 无 ivcore11 device 后端, codegen 硬绑 LLVM NVPTX + ptxas, 无 OpenCL/native fallback。
2. **让包能加载以在库层复现壁垒(Failure Gate)**
   - 首次 `Pkg.instantiate` 预编译崩溃: `UndefVarError: ptxas not defined in CUDA_Compiler_jll`(CoreX 驱动 10.2 使 JLL 平台标签为 `cuda="none"`, 不提供 ptxas)。
   - 先试官方修复: pin `[CUDA_Runtime_jll] version="12.6"` 以获取 cuda+12 的 ptxas 制品 → **失败**: 该 CUDA_Runtime v0.23.0 cuda+12.6 制品在本环境不可达(NJU 镜像 HTTP 404, GitHub release 回退被限速 <1 B/s, resolve 报错)。
   - 改用 repo-local 守卫: `CUDACore/src/precompile.jl` 给 `@compile_workload` 加 `_corex_ptxas_available()` 探针, 无 ptxas 时跳过该 workload → **预编译成功**(`build/precompile_resilient.log` exit=0)。
3. **库层复现**: `using CUDA` 加载成功但 `CUDA.functional()=false`; `functional(true)` 抛 `NVIDIA driver too old`(initialization.jl:92-96, 因驱动报 CUDA 10.2 < 12); `CUDA.devices()` 抛 `CUDA_ERROR_NOT_INITIALIZED(3)`; 平凡 kernel `CuArray .+= 1` 抛 `NVIDIA driver too old`。

## 结果
- **compile_status = success**: 经上述 repo-local 绕过后, Julia 包(CUDA + CUDACore + 全部 lib 子包 + 测试工程)均预编译成功。
- **test_status = not_attempted / tests_run = 0**: 完整测试套件 `julia --project=test test/runtests.jl` 在 `test/setup.jl:14` `@assert CUDA.functional(true)` 处立即崩溃(`ERROR: LoadError: NVIDIA driver too old`), 在任何运行时用例被 include/执行之前即中止。属 terminal 初始化壁垒导致整套无法启动, **非**挑子集/漏报; 完整输出见 `test/test.log`。
- **overall_status = blocked**: 存在未解决的 terminal blocker。

## Failure Gate 分类
- **Terminal #1**: ivcore11 驱动拒绝 NVIDIA PTX 文本(`cuModuleLoadData`→200)。CUDA.jl 只能产 NV PTX 且无 ivcore11 后端, 不可调和; 不允许改 /usr/local/corex。已用裸探针给出证据。
- **Terminal #2**: CUDA.jl 6.x 要求驱动 CUDA>=12, CoreX 报告 10.2。驱动版本由 SDK 决定不可改; 即便 repo-local 改掉版本门禁, 下游仍撞 #1。
- **Workaround-able #3(已解决)**: 预编译 `ptxas` 缺失崩溃 → precompile.jl 守卫绕过, 预编译通过。官方 version-pin 修复因制品不可达而无法采用(已记录)。

## 复现命令
```bash
. /etc/profile.d/github-migration.sh; . /etc/profile.d/corex.sh
export PATH=/opt/julia-1.11.5/bin:$PATH JULIA_DEPOT_PATH=/home/repos/.julia_depot

# 驱动探针(terminal 壁垒证据)
cd corex_port/probe
clang ptx_probe.c -I$COREX_PATH/include -L$COREX_PATH/lib -lcuda -o ptx_probe && ./ptx_probe
clang++ -x cuda --cuda-gpu-arch=ivcore11 native_kernel.cu -I$COREX_PATH/include -L$COREX_PATH/lib -lcudart -lcuda -o native_kernel && ./native_kernel

# 库层复现
cd /home/repos/CUDA.jl
julia --project=. -e 'using Pkg; Pkg.precompile()'
julia --project=. -e 'using CUDA; @show CUDA.functional(); CUDA.functional(true)'
julia --project=test test/runtests.jl   # 在 setup.jl:14 崩溃, 0 用例执行
```
