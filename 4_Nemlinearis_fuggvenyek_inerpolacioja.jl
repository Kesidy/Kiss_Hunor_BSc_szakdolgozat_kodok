using LinearAlgebra
using Flux
using Flux: @epochs
using LinearAlgebra
using PyPlot
pygui(true)
using Random
using BSON: @save, @load

###############################Adatok betöltése###########################

@load("THE_data_NEMLIN.bson")



###############################Neurális háló topológiájának és paramétereinek megadása###########################

#Háló felépítése
model_be =  Chain(
    Conv((10,), 1 => 2, stride=4,pad=(4,), leakyrelu),
    Dropout(0.25),
    Conv((6, ), 2 => 3, stride=2, pad = (2,), leakyrelu),
    Dropout(0.25),
    x->reshape(x,45,:),
    Dense(45,2))

model_ki= Chain(
    Dense(2,45),
    x->reshape(x,15,3,:),
    BatchNorm(3,leakyrelu),
    ConvTranspose((6, ), 3 => 2, stride=2, pad = 2, leakyrelu),
    BatchNorm(2,leakyrelu),
    ConvTranspose((10,), 2 => 1, stride=4, pad = 3, leakyrelu),
    x->reshape(x,120,:),
    Dense(120,120, leakyrelu),
    (x->reshape(x,120,1,:)))

#Háló tesztelés

#Bemeneti háló tesztelése
t0=train_data
t1=model_be.layers[1](FFT_nemlin_norm)
t2=model_be.layers[2](t1)
t3=model_be.layers[3](t2)
t4=model_be.layers[4](t3)
t5=model_be.layers[5](t4)
t5=model_be.layers[6](t5)

#Kimeneti háló tesztelése
test1=model_ki.layers[1](t5)
test2=model_ki.layers[2](test1)
test3=model_ki.layers[3](test2)
test4=model_ki.layers[4](test3)
test5=model_ki.layers[5](test4)
test6=model_ki.layers[6](test5)
test7=model_ki.layers[7](test6)
test8=model_ki.layers[8](test7)
test9=model_ki.layers[9](test8)

#Hiba függvény definiálása
loss(x) = Flux.mse(model_ki(model_be(x)),x)



#Hiba követő vektorok definiálása
loss(train_data)
loss(test_data)
loss_train=Vector{Float32}(undef,0)
loss_test=Vector{Float32}(undef,0)

#Hiba számítás definiálása
function evalcbc()
    push!(loss_train,loss(train_data))
    push!(loss_test,loss(test_data))
    if loss_test[end]<loss_test[end-1]
        @save "final_test.bson" model_be model_ki
    end
    @show loss_train[end]
    @show loss_test[end]
end
throttled_cb()=Flux.throttle(evalcbc(), 5)



#Optimalizáló megadása
opt = ADAM(0.002)


#Tanítás
begin
    ps = params(model_ki, model_be)
    # Flux.testmode!(model_be,false)
    # testmode!(model_ki,false)
    @progress "Training" for i in 1:5000
        Flux.train!(loss, ps, (train_data,), opt, cb = throttled_cb) #, cb = throttle(evalcb, 10)
    end
end
###############################Kiértékelés###########################

@load("final_test.bson")
Flux.testmode!(model_ki,true);
Flux.testmode!(model_be,true);

#Globális tesztmód bekapcsolása
Flux.testmode!(model_ki,true);
Flux.testmode!(model_be,true);


####################Epszilon interpolárás#####################

myinterp(θ,p1,p2)=(1-θ)*p1+ θ*p2


begin
    szamok=[140,250] #két interpolárt FFT megadása
    interpolációs_fok=ip_fok=0.5 #interpolációs fog megadása


    E_ip=myinterp(ip_fok,eps_full[szamok[1]],eps_full[szamok[2]])
    FFT_ip = get_FFT_ToTrain(E_ip)
    FFT_ip_min, FFT_ip_max = extrema(FFT_ip)
    FFT_ip_norm=transf.(FFT_ip,FFT_ip_max,FFT_ip_min)
    THE_FFT_ip_norm=cat(FFT_ip_norm, dims=3)


    latents=model_be(FFT_nemlin_norm[:,:,szamok]) #számok sorszámú FFT-eket kódolón átviszem
    latent_ip=myinterp(interpolációs_fok,latents[:,1],latents[:,2]) #interpoláom azt átvittem FFT-eket
    ki=model_ki(latents) #nem interpolárt FFT-eket átviszem a dekódolón
    ki_ip=model_ki(latent_ip)

    begin
        figure("Epszilon interpoláció"); clf()
            title("Interpolált FFT normalizált alakban")
            xlabel(L"\mathrm\omega [-]")
            ylabel("FFT [-]");
            ax=gca()
            ax.spines["top"].set_visible(false)
            ax.spines["right"].set_visible(false)
            plot((ωs_plot_norm),reshape(THE_FFT_ip_norm,120),label="Pontos köztes FFT")
            plot(ωs_plot_norm,reshape(ki_ip,120),label="NN-el interpolált FFT")
            plot(ωs_plot_norm,ki[:,:,1],label="FFT1")
            plot(ωs_plot_norm,ki[:,:,2],label="FFT1")
            legend()
    end
end


begin
    FFT_modell_full=model_ki(model_be(THE_FFT_ip_norm))
    figure("Nemlin illesztés összehasonlító"); clf()
    title("Nemlinerási szimuláció illesztési összehasonlító normalizált formában")
    xlabel(L"\mathrm\omega [-]")
    ylabel("FFT [-]");
    ax=gca()
    ax.spines["top"].set_visible(false)
    ax.spines["right"].set_visible(false)
    plot((ωs_plot_norm),reshape(THE_FFT_ip_norm,120),label="Pontos FFT")
    plot((ωs_plot_norm),reshape(FFT_modell_full,120),label="Közelített FFT")
    legend()
end


FFT1_1t=eps_full[szamok[1]];
FFT1_2t=eps_full[szamok[2]];
FFT_1ss_epszilon=getindex(FFT1_1t,1) #1. epszilon
FFT_1ss_epszilon=getindex(FFT1_2t,1) #2. epszilon
E_ip[1] #közelített epszilon
##################Latens space-ek ábárzolása###########
latents_full=model_be(FFT_nemlin_norm) #Össze FFT latens space generálás

xdata=latents_full[1, :] #Latens space X kord.
ydata=latents_full[2, :] #Latens space Y kord.
zdata=epszilon_values=eps_full #Epszilonok megadása

#1-es epszilon latent space adatainak kikérése
eps1_latent_space_x=xdata[szamok[1]];
eps1_latent_space_y=ydata[szamok[1]];

#2-es epszilon latent space adatainak kikérése
eps2_latent_space_x=xdata[szamok[2]];
eps2_latent_space_y=ydata[szamok[2]];

#interpolált epszilon
eps_int_x=myinterp(ip_fok,eps1_latent_space_x,eps2_latent_space_x);
eps_int_y=myinterp(ip_fok,eps1_latent_space_y,eps2_latent_space_y);

begin
    figure("xy"); clf()
    title("250 DB nemlineáris FFT látens térének ábrázolása, hozzájuk tartozó epszilon értékek színskálázva")
    xlabel("Látens tér első dimeziójának értékei [-]");
    ylabel("Látens tér második dimenziójának értékei [-]");
    ax=gca()
    ax.spines["top"].set_visible(false)
    ax.spines["right"].set_visible(false)
    scatter(xdata,ydata,c=zdata,label="Látens tér x-y")
    colorbar()
    scatter(eps1_latent_space_x,eps1_latent_space_y,label=L"\mathrm\varepsilon1",alpha=1, color="red")
    scatter(eps2_latent_space_x,eps2_latent_space_y,label=L"\mathrm\varepsilon2", color="#e87c00")
    scatter(eps_int_x,eps_int_y,label=L"\mathrm\varepsilon", color="black")
    legend()
end

# begin
#     figure("x"); clf()
#     title("500 DB FFT látens térének első dimenziójának értékei")
#     xlabel("Epszilon [-]");
#     ylabel("Látens tér tanult értékei [-]");
#     ax=gca()
#     ax.spines["top"].set_visible(false)
#     ax.spines["right"].set_visible(false)
#     plot(zdata,xdata,label="latent space x")
#     legend()
# end
#
# begin
#     figure("y"); clf()
#     title("500 DB FFT látens térének második dimenziójának értékei")
#     xlabel("Epszilon [-]");
#     ylabel("Látens tér tanult értékei [-]");
#     ax=gca()
#     ax.spines["top"].set_visible(false)
#     ax.spines["right"].set_visible(false)
#     plot(zdata,ydata,label="latent space y")
#     legend()
# end
