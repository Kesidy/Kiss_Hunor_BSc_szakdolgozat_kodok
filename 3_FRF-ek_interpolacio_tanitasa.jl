using LinearAlgebra
using Flux
using Flux: @epochs
using LinearAlgebra
using PyPlot
pygui(true)
using Random
using BSON: @save, @load

###############################Adatok struktúrálása###########################

@load("THE_data.bson")


#Bemeteni adatok struktúrálása
inputt=ωs;
THE_data;

#Adatok szétbontása teszt és train halmazokra
begin
    idxs=shuffle(1:500)
    idxs1=idxs[1:250]
    idxs2=idxs[251:500]
end

begin
    THE_dataR=reshape(THE_data,120,500)
    train_data0=[getindex(THE_dataR,:,idxs1[i]) for i in 1:250]
    test_data0=[getindex(THE_dataR,:,idxs2[i]) for i in 1:250]
end

begin
    train_data=zeros(120,1,250)
    for i in eachindex(train_data0)
        train_data[:,1,i] .= train_data0[i]
    end

    test_data=zeros(120,1,250)
    for i in eachindex(test_data0)
        test_data[:,1,i] .= test_data0[i]
    end
end
begin
    train_data=zeros(120,1,250)
    for i in eachindex(train_data0)
        train_data[:,1,i] .= train_data0[i]
    end

    test_data=zeros(120,1,250)
    for i in eachindex(test_data0)
        test_data[:,1,i] .= test_data0[i]
    end
end

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
t1=model_be.layers[1](train_data)
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
    @progress "Training" for i in 1:200
        Flux.train!(loss, ps, (train_data,), opt, cb = throttled_cb) #, cb = throttle(evalcb, 10)
    end
end
###############################Kiértékelés###########################

#Hibafüggvény ábrázolása
@load "final_test.bson" model_be model_ki
begin
figure("Hibafüggvények"); clf()
        title("Hibafüggvények")
        xlabel("Epoch [-]");
        ylabel("Hibafüggvény [-]");
        ax=gca()
        ax.spines["top"].set_visible(false)
        ax.spines["right"].set_visible(false)
        plot(2:2:2*length(loss_train),loss_train,label="Tanított adatok hibája")
        plot(2:2:2*length(loss_test),loss_test,label="Teszt adatok hibája")
        legend()
end


#FRF illesztési jóság ellenőrzése
begin
    Flux.testmode!(model_ki,true)
    Flux.testmode!(model_be,true)
    begin
        ki_train=model_ki(model_be(train_data))[:,:,[175]] #egy train FRF végig vitele az NN-en
        ki_test=model_ki(model_be(test_data))[:,:,[109]] #egy tesz FRF végig vitele az NN-en
        end
            begin
                begin
                figure("Train"); clf()
                    title("Normalizált nagyítás függvény a teszt adatbázisból")
                    xlabel("Normalizált körfrekvencia [-]");
                    ylabel("Normalizált nagyítás [-]");
                    ax=gca()
                    ax.spines["top"].set_visible(false)
                    ax.spines["right"].set_visible(false)
                    plot((inputt),train_data[:,1,175],label="Pontos nagyítás függvény")
                    plot((inputt),ki_train[:,1,1],label="Közelített nagyítás függvény")
                    legend()
                end
                begin
                figure("Test"); clf()
                    title("Normalizált nagyítás függvény a tanításra szánt adatbázisból")
                    xlabel("Normalizált körfrekvencia [-]");
                    ylabel("Normalizált nagyítás [-]");
                    ax=gca()
                    ax.spines["top"].set_visible(false)
                    ax.spines["right"].set_visible(false)
                    plot((inputt),test_data[:,1,109],label="Pontos nagyítás függvény")
                    plot((inputt),ki_test[:,1,1],label="Közelített nagyítás függvény")
                    legend()
                end
            end
end

#Globális tesztmód bekapcsolása

Flux.testmode!(model_ki,true);
Flux.testmode!(model_be,true);


####################Tömeg interpolárás#####################

myinterp(θ,p1,p2)=(1-θ)*p1+ θ*p2


begin
    szamok=[200,300] #két interpolárt FRF megadása
    interpolációs_fok=ip_fok=0.5 #interpolációs fog megadása
    H(ω) = inv(-ω^2*M_ip+im*ω*C+K)
    begin
        M_ip=myinterp(ip_fok,Ms[szamok[1]],Ms[szamok[2]])
        C = [21 -11;-11 11]
        K = [6800 -3800;-3800 3800]
        H(ω) = inv(-ω^2*M_ip+im*ω*C+K)
        Q=1;q=1
        transf(H,Hmax,Hmin) = ((H) - Hmin)/(Hmax-Hmin)

        #Fázisszög adat
        ωs0 = LinRange(0.,60,120);
        ωmin,ωmax = extrema((ωs0));
        ωs = transf.(ωs0,Ref(ωmax),Ref(ωmin));
        Hωs = H.(ωs0);

        #Nagyítás transzformálás

        absHωs0 = abs.(getindex.(Hωs,1,1));
        Hωmin,Hωmax = extrema((absHωs0));
        absHωs = transf.((absHωs0),Ref(Hωmax),Ref(Hωmin));
        #angHωs = angle.(getindex.(Hωs,Q,q));

        #Fázisszög tranformálás
        angHωs0 = angle.(getindex.(Hωs,Q,q));
        Hωmin,Hωmax = extrema((angHωs0));
        angHωs = transf.((angHωs0),Ref(Hωmax),Ref(Hωmin));

    end

    THE_data_ip=cat(absHωs, dims=3)


    latents=model_be(THE_data[:,:,szamok]) #számok sorszámú FRF-eket kódolón átviszem
    latent_ip=myinterp(interpolációs_fok,latents[:,1],latents[:,2]) #interpoláom azt átvittem FRF-eket
    ki=model_ki(latents) #nem interpolárt FRF-eket átviszem a dekódolón
    ki_ip=model_ki(latent_ip)
    nem_interpolárt=model_ki(model_be(THE_data_ip))
    begin
        figure("Tömeg interpoláció"); clf()
            title("Interpolált FRF normalizált alakban")
            xlabel("ω [-] normalizált");
            ylabel("Normalizált nagyítás [-]");
            ax=gca()
            ax.spines["top"].set_visible(false)
            ax.spines["right"].set_visible(false)
            plot((inputt),reshape(THE_data_ip,120),label="Pontos köztes FRF")
            # plot(inputt,reshape(nem_interpolárt,120),label="Tanított, de nem interpolárt FRF1+2")
            plot(inputt,reshape(ki_ip,120),label="NN-el interpolált FRF")
            plot(inputt,ki[:,:,1],label="FRF1")
            plot(inputt,ki[:,:,2],label="FRF2")
            legend()
    end
end

FRF1_1t=Ms[szamok[1]];
FRF2_1t=Ms[szamok[2]];
FRF1_1ss_weight=getindex(FRF1_1t,1) #1. tömeg
FRF2_1ss_weight=getindex(FRF2_1t,1) #2. tömeg
M_ip[1] #közelített tömeg
##################Latens space-ek ábárzolása###########
latents_full=model_be(THE_data) #Össze FRF latens space generálás

xdata=latents_full[1, :] #Latens space X kord.
ydata=latents_full[2, :] #Latens space Y kord.
zdata=mass_values=[m for m in LinRange(1,10,500)] #Összes tömeg ismét legenerálása

#1-es tömeg latent space adatainak kikérése
m1_latent_space_x=xdata[szamok[1]];
m1_latent_space_y=ydata[szamok[1]];

#2-es tömeg latent space adatainak kikérése
m2_latent_space_x=xdata[szamok[2]];
m2_latent_space_y=ydata[szamok[2]];

#interpolált tömege
m_int_x=myinterp(ip_fok,m1_latent_space_x,m2_latent_space_x);
m_int_y=myinterp(ip_fok,m1_latent_space_y,m2_latent_space_y);

begin
    figure("xy"); clf()
    title("500 DB FRF látens térének ábrázolása, hozzájuk tartozó tömegei [kg] színskálázva")
    xlabel("Látens tér első dimeziójának értékei [-]");
    ylabel("Látens tér második dimenziójának értékei [-]");
    ax=gca()
    ax.spines["top"].set_visible(false)
    ax.spines["right"].set_visible(false)
    scatter(xdata,ydata,c=zdata,label="Látens tér x-y")
    colorbar()
    scatter(m1_latent_space_x,m1_latent_space_y,label="m_1_1",alpha=1, color="red")
    scatter(m2_latent_space_x,m2_latent_space_y,label="m_1_2", color="#e87c00")
    scatter(m_int_x,m_int_y,label="m_interpolált", color="black")

    legend()
end

# begin
#     figure("x"); clf()
#     title("500 DB FRF látens térének első dimenziójának értékei")
#     xlabel("Tömeg [kg]");
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
#     title("500 DB FRF látens térének második dimenziójának értékei")
#     xlabel("Tömeg [kg]");
#     ylabel("Látens tér tanult értékei [-]");
#     ax=gca()
#     ax.spines["top"].set_visible(false)
#     ax.spines["right"].set_visible(false)
#     plot(zdata,ydata,label="latent space y")
#     legend()
# end
