# DeconvOptim.jl (branch new_methods) on the stacks written by benchmark.py.
#
#   julia -t 4 --project=<DeconvOptim.jl checkout> benchmark.jl OUT NZ NY NX [solvers...]
#
# Every solver runs twice: the first call includes Julia's compilation, the second is the warm time.
using DeconvOptim, FFTW, Statistics, Printf

FFTW.set_num_threads(4)

const BACKGROUND, GAMMA, ITER = 5f0, 1f-4, 30
out = ARGS[1]
nz, ny, nx = parse.(Int, ARGS[2:4])
wanted = length(ARGS) > 4 ? ARGS[5:end] : ["map_lean", "rl", "admm_tv", "lbfgs_tv"]
tag = "$(nz)x$(ny)x$(nx)"
# a (z, y, x) C-order file is an (x, y, z) Julia array
rd(name) = Array(reshape(reinterpret(Float32, read(joinpath(out, "$(tag)_$(name).f32"))), (nx, ny, nz)))
measured, psf = rd("measured"), rd("psf")
pad = (0.0, 0.0, 0.25)                 # along z only, as in benchmark.py
bg_norm = BACKGROUND / mean(measured)  # deconvolution() normalises the data to mean 1

solvers = Dict(
    "map_lean" => () -> map_cj_lean(measured, psf; gamma=GAMMA, iterations=ITER, background=BACKGROUND, padding=pad),
    # the other data terms and the Hessian penalty (test_deconvolution.py); the recipe multiplies
    # gamma by 2 * (mean - background) for the Gauss loss
    "map_lean_anscombe" => () -> map_cj_lean(measured, psf; loss=:anscombe, gamma=GAMMA, iterations=ITER,
                                             background=BACKGROUND, padding=pad),
    "map_lean_gauss" => () -> map_cj_lean(measured, psf; loss=:gauss, gamma=GAMMA * 2 * max(mean(measured) - BACKGROUND, 1),
                                          iterations=ITER, background=BACKGROUND, padding=pad),
    "map_lean_hessian" => () -> map_cj_lean(measured, psf; penalty=:hessian_frobenius, gamma=10 * GAMMA, iterations=ITER,
                                            background=BACKGROUND, padding=pad),
    "rl" => () -> richardson_lucy_iterative(measured, psf; regularizer=nothing, iterations=ITER),
    "admm_tv" => () -> first(deconvolution(measured, psf; regularizer=TV(num_dims=3), λ=0.01f0, background=bg_norm,
                                           opt=ADMM(iterations=50), padding=pad)),
    "lbfgs_tv" => () -> first(deconvolution(measured, psf; regularizer=TV(num_dims=3), λ=0.01f0, background=bg_norm,
                                            iterations=20, padding=pad)),
    # the setting shown in the result figure: a TV weight that suits the sparse test object
    "lbfgs_tv60" => () -> first(deconvolution(measured, psf; regularizer=TV(num_dims=3), λ=1f-4, background=bg_norm,
                                              iterations=60, padding=pad)),
)

for name in wanted
    first_s = @elapsed solvers[name]()
    warm_s = @elapsed result = solvers[name]()
    write(joinpath(out, "$(tag)_jl_$(name).f32"), Float32.(result))
    @printf("RESULT {\"shape\": \"%s\", \"solver\": \"jl_%s\", \"seconds\": %.2f, \"first_call_s\": %.2f, \"maxrss_mb\": %.0f}\n",
            tag, name, warm_s, first_s, Sys.maxrss() / 2^20)
    flush(stdout)
end
