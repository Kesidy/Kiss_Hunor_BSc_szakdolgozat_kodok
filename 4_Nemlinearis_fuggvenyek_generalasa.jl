using DifferentialEquations, FFTW
## Normalizált FFT számító
function getFFT(ts,usnoise)
    ωs = 2π.*(0:(length(ts)-1))./(ts[end]-ts[1]);
    n_fft = fft(usnoise)[1:length(ωs)÷2];
    # n_fft = n_fft[1:length(n_fft)÷2];
    n_fft .= n_fft./(length(usnoise)/2)
    return collect(ωs[1:length(n_fft)]), n_fft
end
function getFFT_fromsol(sol, Δt, T; idxs=1)
    ts = 0:Δt:T;
    u_idx = sol.(ts, idxs = idxs)
    ωs, _fft = getFFT(ts,u_idx)
end

function f(du,u,p,t)
    ζ, ε = p
    du[1] = u[2];
    du[2] = -2ζ*u[2] - (one(eltype(u)) + ε*u[1]^2)*u[1] #+ sin(t)
end

function get_FFT_ToTrain(ε;
                T = 43*2π, Δt = 2π/100, dt = 2π/1000, ζ = 0.05, ω_limit = 2.75)
    # dt: szimulációhoz időlépés
    # Δt: mintavételezési idő az FFThez
    # ω_limit: max ω
    tspan = (0,T)
    u₀ = [0.,1.];
    prob = ODEProblem(f,u₀,tspan,(ζ,ε)) # ζ = 0.05
    sol = solve(prob, Tsit5(), dt = dt, adaptive=false);
    @assert sol.retcode == :Success "A megadott ε = $(ε) instabil"
    _ωs, _fft = getFFT_fromsol(sol,Δt,T);
    uptoidx = searchsortedfirst(_ωs,ω_limit);
    return abs.(_fft[1:uptoidx])
end


# T = 43*2π;
#  Δt = 2π/100;
#  dt = 2π/1000;
#  ζ = 0.05;
#  ω_limit = 2.75;
# tspan = (0,T)
# u₀ = [0.,1.];
# prob = ODEProblem(f,u₀,tspan,(ζ,0.413)) # ζ = 0.05
# sol = solve(prob, Tsit5(), dt = dt, adaptive=false);
# @assert sol.retcode == :Success "A megadott ε = $(ε) instabil"
# _ωs, _fft = getFFT_fromsol(sol,Δt,T);
# uptoidx = searchsortedfirst(_ωs,ω_limit);
# return abs.(_fft[1:uptoidx])
# ωs_plot = _ωs[1:uptoidx]


## Alkalmazás:
using PyPlot, LaTeXStrings
pygui(true); PyPlot.rc("text", usetex=true);
py_colors=PyPlot.PyDict(PyPlot.matplotlib."rcParams")["axes.prop_cycle"].by_key()["color"];


Afft1 = get_FFT_ToTrain(0.525)
Afft2 = get_FFT_ToTrain(0.55)
Afft3 = get_FFT_ToTrain(0.575)



begin
    figure(3); clf();

    plot(ωs_plot, Afft1)
    plot(ωs_plot, Afft2)
    plot(ωs_plot, Afft3)
    xlabel(L"\omega",fontsize=16); ylabel(L"|\mathrm{FFT}\{x\}(\omega)|",fontsize=16);
    # ylim(bottom = -0.7,top = 0.7)
end

eps_full = LinRange(-0.6,0.6,250) #eplszilok mintavételezése

sol_eps_full = get_FFT_ToTrain.(eps_full)

FFT_nemlin=zeros(120,1,250)
for i in eachindex(sol_eps_full)
    FFT_nemlin[:,1,i] .= sol_eps_full[i]
end
FFT_nemlin

####################Normalizálás
transf(val,valmax,valmin) = ((val) - valmin)/(valmax-valmin)
ωs_plot_max, ωs_plot_min = extrema((ωs_plot))
ωs_plot_norm = transf.(ωs_plot,Ref(ωs_plot_max),Ref(ωs_plot_min))

FFT_nemlin_szelso=FFT_nemlin_min,FFT_nemlin_max=[extrema((indx)) for indx in sol_eps_full]

begin
    MinimumFFT=zeros(250)
    for i in eachindex(MinimumFFT)
        MinimumFFT[i] = getindex(getindex(FFT_nemlin_szelso,i),1)
    end
end
MinimumFFT

begin
    MaximumFFT=zeros(250)
    for i in eachindex(MaximumFFT)
        MaximumFFT[i] = getindex(getindex(FFT_nemlin_szelso,i),2)
    end
end
MaximumFFT

FFT_nemlin_norm_vector=[(((getindex(FFT_nemlin,:,:,i)) .- getindex(MinimumFFT,i))./(getindex(MaximumFFT,i)-getindex(MinimumFFT,i))) for i in 1:250]

FFT_nemlin_norm=zeros(120,1,250)
begin
    for i in eachindex(FFT_nemlin_norm_vector)
        FFT_nemlin_norm[:,:,i] = FFT_nemlin_norm_vector[i]
    end
end
FFT_nemlin_norm

begin
    figure("FTT 250DB"); clf()
    plot(ωs_plot_norm,FFT_nemlin_norm[:,:,1],label="FFT")
    legend()
end

@save("THE_data_NEMLIN.bson",FFT_nemlin_norm, ωs_plot_norm, eps_full, get_FFT_ToTrain, transf)
