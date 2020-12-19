using LinearAlgebra
using Flux
using Flux: @epochs
using JLD2, FileIO

@load "FRF2x2.jld2" absHωs angHωs ωs Hωmin Hωmax angmax angmin #Generált adatok betöltése

half_input=[[ω] for ω in ωs]; #x/2
input=zip(half_input,half_input); #x
input=[[ω] for ω in ωs]; #x/2


m_data_absHωs=[[elem] for elem in absHωs]; #y1
m_data_angHωs=[[part] for part in angHωs]; #y2
m_data=zip(m_data_absHωs,m_data_angHωs); #y
m_data=[[absHωs[i],angHωs[i]] for i in eachindex(angHωs)]; #y2

#Neurális hálózat topológiájának definiálása
# ntwk=Chain(Dense(1, 4, σ), Dense(4, 4, tanh), Dense(4,2)); #1. verzió
ntwk=Chain(Dense(1, 20, leakyrelu), Dense(20, 20, tanh), Dense(20, 20, σ), Dense(20,2,σ)); #2. verzió
# ntwk=Chain(Dense(1, 5, leakyrelu), Dense(5, 6, leakyrelu), Dense(6, 6, leakyrelu), Dense(6, 5, leakyrelu), Dense(5,2, tanh), Dense(2,2, σ)); #3. verzió

loss(x, y) = Flux.mse(ntwk(x), y); #Hibafüggvény def.
opt = ADAM(0.002); #Optimalizálási mód és paraméter megadása
ps = Flux.params(ntwk);

#Tanítás
begin
    @progress "Training" for i in 1:500
        Flux.train!(loss, ps, zip(input,m_data), opt) #, cb = throttle(evalcb, 10)
    end
end

#Ábrázolás
using PyPlot
pygui(true)
results=ntwk.(input);
begin
    figure("Nagyítás+fázisszög közelítés"); clf()
    title("Nagyítás és fázisszög közelítése normalizált formában")
    xlabel("ω [-]");
    ylabel("|H(ω)| [-]
    φ(ω) [-]");
    ax=gca()
    ax.spines["right"].set_visible(false)
    ax.spines["top"].set_visible(false)
    plot(ωs,getindex.(m_data,1),label="|H(ω)|")
    plot(ωs,getindex.(m_data,2),label="φ(ω)")

    results=ntwk.(input);
    plot(ωs,getindex.(results,1),label="Közelített nagyítás")
    plot(ωs,getindex.(results,2),label="Közelített fázisszög")
    ax.legend()
end

begin
    #Eredmények vektoroztatása
    H_abs_result=getindex.(results,1)
    H_angle_result=getindex.(results,2)
    #Inverz fv. definiciója
    invtranf(H_result, Hmax, Hmin)= H_result*(Hmax-Hmin)+Hmin
    #2 eredmény invertálása
    H_inv_plot=invtranf.(H_abs_result,Hωmax,Hωmin);
    H_inv_plot_angle=invtranf.(H_angle_result,angmax,angmin);
end

#Inverz plot nagyítás
begin
    figure("Nagyítás közelítésének ábrázolásae"); clf()
    title("Nagyítás közelítésének ábrázolása")
    xlabel("ω [rad]");
    ylabel("|H(ω)|");
    ax=gca()
    ax.spines["right"].set_visible(false)
    ax.spines["top"].set_visible(false)
    plot(ωs0,H_inv_plot,label="Közelített nagyítás")
    plot(ωs0,absHωs0,label="|H(ω)|")
    ax.legend()
end

#Inverz plot fázisszög
begin
    figure("Fázisszög közelítésének ábrázolása"); clf()
    title("Fázisszög közelítésének ábrázolása")
    xlabel("ω [rad]");
    ylabel("φ(ω) [rad]");
    ax=gca()
    ax.spines["right"].set_visible(false)
    ax.spines["top"].set_visible(false)
    plot(ωs0,H_inv_plot_angle,label="Közelített fázisszög")
    plot(ωs0,angHωs0,label="φ(ω)")
    ax.legend()
end


max_orginal=maximum(absHωs0)
max_learn=maximum(H_inv_plot)
max_loss_=(maximum(absHωs0)-maximum(H_inv_plot))/(maximum(absHωs0))
max_loss_percent=(maximum(absHωs0)-maximum(H_inv_plot))/(maximum(absHωs0))*100
