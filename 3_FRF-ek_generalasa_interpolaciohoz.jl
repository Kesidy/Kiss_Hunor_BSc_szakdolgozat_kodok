########################################FRF-ek generálása##############
using PyPlot
pygui(true)
using BSON: @save, @load

H(ω,M,C,K)=(-ω^2*M + im*ω*C + K)^(-1)
transf(val,valmax,valmin) = ((val) - valmin)/(valmax-valmin)

Ms=[[m 0.;0. 1.1] for m in LinRange(1,10,500)]

C = [21 -11;-11 11]
K = [6800 -3800;-3800 3800]

ωs0 = LinRange(0.,60,120)
ωmin,ωmax = extrema((ωs0))
ωs = transf.(ωs0,Ref(ωmax),Ref(ωmin))

Hs=[H.(ωs0,Ref(M),Ref(C),Ref(K)) for M in Ms]
H11s=[getindex.(h,1,1) for h in Hs]
absH11s0=[abs.(h) for h in H11s]



asd=absH11s0min,absH11s0max = [extrema((absh)) for absh in absH11s0]
###########################################################################
begin
    Minimum=zeros(500)
    for i in eachindex(Minimum)
        Minimum[i] = getindex(getindex(asd,i),1)
    end
end

Minimum

begin
    Maximum=zeros(500)
    for i in eachindex(Maximum)
        Maximum[i] = getindex(getindex(asd,i),2)
    end
end
Maximum

#################################################
# begin
#     qwerty=zeros(120)
#     for i=1:120
#         qwerty[i] = transf.((absH11s0[i]),Ref(Maximum[i]),Ref(Minimum[i]))
#     end
# end
# qwerty


H_data=zeros(length(ωs0),1,length(absH11s0))

for i in eachindex(absH11s0)
    H_data[:,1,i] .= absH11s0[i]
end

H_data

# daf=zeros(length(ωs0),1,length(absH11s0))
# begin
#     for i=(1:(500*120))
#         daf[i,1,] .= transf.((H_data[120,:,i]),Ref(Maximum[i]),Ref(Minimum[i]))
#     end
# end
# daf
#
#
#
# transf(val,valmax,valmin) = ((val) - valmin)/(valmax-valmin)
bob=[(((getindex(H_data,:,:,i)) .- getindex(Minimum,i))./(getindex(Maximum,i)-getindex(Minimum,i))) for i in 1:500]


THE_data=zeros(length(ωs0),1,length(bob))

for i in eachindex(bob)
    THE_data[:,:,i] = bob[i]
end
THE_data

begin
    figure("FRF 500DB"); clf()
    plot(ωs,THE_data[:,:,1],label="FRF")
    legend()
end


using BSON: @save, @load

@save("THE_data.bson",THE_data, ωs, Ms)
