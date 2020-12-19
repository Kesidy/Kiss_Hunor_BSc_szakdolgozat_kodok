using LinearAlgebra
using Flux
using Flux: @epochs
using Base.Iterators: repeated

#Adat előkészítés
φs=LinRange(0.,2π,500); #tartomány,mintevételezés nagysága
input=[[φ] for φ in φs] #x strukturálás
m_data=[[sin(φ)] for φ in φs] #y strukturálás

#Hálózat topológiája
ntwk=Chain(Dense(1, 5, σ), Dense(5, 7, tanh), Dense(7,1));

loss(x, y) = Flux.mse(ntwk(x), y) #Hibafüggvény def.
opt = ADAM(0.0001); #Optimalizálási mód megadása
ps = Flux.params(ntwk);

@epochs 200 Flux.train!(loss, ps, zip(input,m_data), opt) #Tanítás indítása

#FV. rajzolás
using PyPlot
pygui(true);

begin
    figure("Szinusz függvény közelítése"); clf()
    title("Szinusz függvény közelítése")
    xlabel("x");
    ylabel("y");
    ax=gca()
    ax.spines["right"].set_visible(false)
    ax.spines["top"].set_visible(false)
    plot(φs,m_data,label="sin(x)")
    plot(φs,getindex.(ntwk.(input),1),label="fˇs(x)")
    ax.legend()
end
