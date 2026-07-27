# `llvm_compat()` requires being able to initialize the NVPTX backend, so we run the
# precompile workload only when that's supported, to be able to load this package also on
# systems where the backend isn't available.
# CoreX (ivcore11) migration: on the CoreX driver (reports CUDA 10.2, below CUDA.jl 6.x's
# CUDA-12 floor) the JLL machinery provisions no `ptxas` (cuda="none"), so `ptxas_compat()`
# below throws `UndefVarError: ptxas not defined in CUDA_Compiler_jll` and precompilation
# crashes before CUDA.jl can even load. Guard the workload with a ptxas-availability probe so
# the package still precompiles/loads (letting __init__ report the real driver wall) when no
# usable ptxas is present. (repo-local workaround; /usr/local/corex untouched.)
_corex_ptxas_available() = try
    CUDA_Compiler.ptxas(); true
catch
    false
end
if :NVPTX in LLVM.backends() && _corex_ptxas_available()
    @compile_workload begin
        # compile a dummy kernel to precompile the GPUCompiler pipeline.
        # this uses the compiler toolchain, but doesn't need a GPU.
        let
            function _precompile_vadd(a)
                i = threadIdx().x
                @inbounds a[i] += 1f0
                return nothing
            end

            llvm_support = llvm_compat()
            ptxas_support = ptxas_compat()
            # `.sm` is `Set{SMVersion}` (with variants); pick the highest baseline
            # entry <= v"7.5" for a portable precompile artifact.
            llvm_sm = argmax(base_version,
                             filter(sm -> sm.feature_set === :baseline &&
                                          base_version(sm) <= v"7.5",
                                    llvm_support.sm))
            llvm_ptx, ptxas_ptx = default_ptx_versions(llvm_support, ptxas_support)

            target = PTXCompilerTarget(; cap=base_version(llvm_sm), ptx=llvm_ptx, debuginfo=true)
            params = CUDACompilerParams(; sm=llvm_sm, ptx=ptxas_ptx)
            config = CompilerConfig(target, params; kernel=true, name=nothing, always_inline=false)

            tt = Tuple{CuDeviceArray{Float32,1,AS.Global}}
            source = methodinstance(typeof(_precompile_vadd), tt)
            job = CompilerJob(source, config)

            # On Julia < 1.12, GPU compilation during precompilation leaks foreign
            # MIs into native compilation, causing LLVM errors
            # (e.g. "Cannot select: intrinsic %llvm.nvvm.membar.sys").
            @static if VERSION >= v"1.12-"
                # Go through the launch-side cache path so both that path and package-image
                # serialization of the generated image are precompiled.
                compile_or_lookup(job)
            end
        end
    end
end

# kernel launch infrastructure
let CUDACompilerJob = CompilerJob{PTXCompilerTarget, CUDACompilerParams}
    precompile(Tuple{typeof(cufunction), typeof(identity), Type{Tuple{Nothing}}})
    precompile(Tuple{typeof(link_kernel), CUDACompilerJob, Vector{UInt8}, String})

    # GPUCompiler 2.0 caching pipeline (specialized for CUDACore's results struct)
    precompile(Tuple{typeof(compile_or_lookup), CUDACompilerJob})
    precompile(Tuple{typeof(GPUCompiler.cached_results), Type{CUDACompilerResults}, CUDACompilerJob})
end

# scalar reference (used by cuBLAS for alpha/beta parameters)
precompile(Tuple{Type{CuRefValue{Float32}}, Float32})
precompile(Tuple{typeof(pool_free), Managed{DeviceMemory}})
