using LinearAlgebra
#Csillapított rendszer differenciálegyenletének mátrix együtthatói
M = [1.7 0;0 1.1] #Tömegmátrix definiálása
C = [21 -11;-11 11] #Csillapítási mátrix definiálása
K = [6800 -3800;-3800 3800] #Merevségi mátrix definiálása

#Frekvenciaválasz függvény meghatározása
H(ω) = inv(-ω^2*M+im*ω*C+K)

#Gerjesztés helye
Q=1

#Válasz helye
q=1

using PyPlot
pygui(true);

#Skálázás, normalizálás
begin
    transf(H,Hmax,Hmin) = ((H) - Hmin)/(Hmax-Hmin)

    #Fázisszög adat
    ωs0 = LinRange(0.,60,500);
    ωmin,ωmax = extrema((ωs0));
    ωs = transf.(ωs0,Ref(ωmax),Ref(ωmin));
    Hωs = H.(ωs0);

    #Nagyítás transzformálás

    absHωs0 = abs.(getindex.(Hωs,Q,q));
    Hωmin,Hωmax = extrema((absHωs0));
    absHωs = transf.((absHωs0),Ref(Hωmax),Ref(Hωmin));
    #angHωs = angle.(getindex.(Hωs,Q,q));

    #Fázisszög tranformálás
    angHωs0 = angle.(getindex.(Hωs,Q,q));
    angmin,angmax = extrema((angHωs0));
    angHωs = transf.((angHωs0),Ref(angmax),Ref(angmin));
end

#Ábrázolás
begin
    figure("FRF: Nagyyítás ábrázolása"); clf();

    title("FRF: Nagyítás ábrázolása")
    xlabel(" ω [rad]");
    ylabel("|H(ω)|");
    ax=gca()
    ax.spines["right"].set_visible(false)
    ax.spines["top"].set_visible(false)

    plot(ωs0,absHωs0);
end
begin
    figure("FRF: Fázisszög ábrázolása"); clf();
    title("FRF: Fázisszög ábrázolása")
    ylabel("atan(Im{H(ω)}/Re{H(ω)}")
    xlabel("ω [rad]");
    ax=gca()
    ax.spines["right"].set_visible(false)
    ax.spines["top"].set_visible(false)

    plot(ωs0,angHωs0);
end

#Adatok exportálása
using JLD2, FileIO
@save "FRF2x2.jld2" absHωs angHωs ωs Hωmin Hωmax angmax angmin
