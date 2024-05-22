import numpy as np
import matplotlib.pyplot as plt
from scipy import sparse
from scipy.stats import pearsonr
import pandas as pd
import csv
import os


##PARÁMETROS

N = 500                                   # numero de neuronas
N2 = int(N/2)

#Conexiones sinápticas
p = 1                                   # probabilidad de elementos no nulos en la matriz de pesos
gsyn = 0.5                                # peso sinaptico inicial

alpha = 0.25                              # regularizador pesos

#Dinámica
dt = 0.1                                  # paso de tiempo (escala de tiempo 10 ms)
itmax = 2000                              # numero de iteraciones (1000 -> 1 sec)              
sigman = 1                                # Noise standard deviation -> ruido en la dinámica
vt = 0.5                                  # Potencial de threshold
b = 0.5                                   # inversa constante de tiempo sinaptica  (escala de tiempo 10 ms)

#Estímulo
itstim = 200                              # tiempo de estimulo
amp_corriente = 6                         # intensidad estímulo


iout = np.linspace(0,N,num=N,endpoint=False).astype('int')
i_graph=np.array((0,1,2,N2+1,N2+2,N2+3))   # indices a graficar                                 # para guardar 10 salidas


#TARGETS
type_target = 'disc' # - cont - gauss - sec - disc
#romega1 = 1                                # cociente frecuencia alta/baja
#romega2 = 5
amp0 = 4                                  # amplitud funciones objetivo


#ENTRENAMIENTO
ftrain = 1                                # fraccón de neuronas a entrenar
nloop  = 16                              # numero de loops, 0 pre-entramiento, ultimo: post-entrenamiento. Poner nloop=2 para no hacer aprendizaje
nloop_train = 10                         #ultimo loop de entrenamiento

cant_seed = 10


## FUNCIONES
def crear_subcarpeta(nombre_carpeta, nombre_subcarpeta):
    
    subcarpeta_path_total = (os.path.join(nombre_carpeta, nombre_subcarpeta))
    if not os.path.exists(subcarpeta_path_total):
        os.makedirs(subcarpeta_path_total)
    
    return subcarpeta_path_total

def crear_carpetas(num_simulacion):

    nombre_carpeta = f"simulacion_{num_simulacion}"
    if not os.path.exists(nombre_carpeta):
        os.makedirs(nombre_carpeta)

    nombre_subcarpeta_act = f"simulacion_{num_simulacion}_ejemplos_actividad"
    nombre_subcarpeta_pesos = f"simulacion_{num_simulacion}_matrices_pesos"

    sub_act = crear_subcarpeta(nombre_carpeta, nombre_subcarpeta_act)
    sub_pesos = crear_subcarpeta(nombre_carpeta, nombre_subcarpeta_pesos)

    return nombre_carpeta, sub_act, sub_pesos

def crear_archivo_parametros(filename_resultados, num_simulacion, nombre_carpeta):
 #file donde guardo los parámetros de la simulación
    data_parametros = {
        'N': [N],
        'p': [p],
        'gsyn': [gsyn],
        'nloop': [nloop],
        'nloop_train':[nloop_train],
        'cant_seed': [cant_seed],
        'dt': [dt],
        'itmax': [itmax],
        'itstim': [itstim],
        'amp_corriente': [amp_corriente],
        'type_target': [type_target],
        'amp0': [amp0],
        'ftrain': [ftrain],
        'alpha': [alpha],
        'sigman': [sigman],
        'vt': [vt],
        'b': [b],
        'results_file': [filename_resultados],
    }


    df = pd.DataFrame(data_parametros)
    filename_parametros = f'simulacion_{num_simulacion}_parametros.csv'
    csv_parametros_path = os.path.join(nombre_carpeta, filename_parametros)
    df.to_csv(csv_parametros_path, index=False)


def generate_target(type_target,romega1, romega2, num_simulacion, nombre_carpeta):

    target=np.zeros((N,itmax))

    amp=np.random.uniform(size=N)*amp0
    phase=np.random.uniform(size=N)*2*np.pi


    if type_target=='disc':

        indices = [i for i in range(N)]
        indices = np.random.permutation(indices) #índices para identificar a que neurona se le asigna cada frecuencia

        
        romega_vec = np.zeros(N)
        
        for i in range(N2):
         
         romega_vec[indices[i]]= romega1
         romega_vec[indices[i+N2]]=romega2

        
        omega=romega_vec*2*np.pi/itmax
    
        for it in range(itmax):
           target[:,it]=amp*np.cos(it*omega+phase) 

        # Crea un DataFrame de pandas con los datos
        data = {'Neurona': range(N), 'Fase': phase, 'Frecuencia': omega, 'romega': romega_vec, 'Amplitud': amp}
        df = pd.DataFrame(data)
        
        # Guarda el DataFrame en un archivo CSV
        nombre_archivo = f'simulacion_{num_simulacion}_targets.csv'
        csv_target_path = os.path.join(nombre_carpeta, nombre_archivo)
        df.to_csv(csv_target_path, index=False)
        

    if type_target=='una_frec':

        omega=2*romega1*np.pi/itmax*np.ones(N)
    
        for it in range(itmax):
           target[:,it]=amp*np.cos(it*omega+phase) 
           
        # Crea un DataFrame de pandas con los datos
        data = {'Neurona': range(N), 'Fase': phase, 'Frecuencia': omega, 'romega': romega1, 'Amplitud': amp}
        df = pd.DataFrame(data)
        
        # Guarda el DataFrame en un archivo CSV
        nombre_archivo = f'simulacion_{num_simulacion}_targets.csv'
        csv_target_path = os.path.join(nombre_carpeta, nombre_archivo)
        df.to_csv(csv_target_path, index=False)
    
            
    if type_target=='cont':       
        omega=(romega1-1)*(np.random.uniform(size=N))+1
        omega=omega*2*np.pi/itmax
        #omega=np.random.permutation(omega)       # random permutation
        for it in range(itmax):
            target[:,it]=amp*np.cos(it*omega+phase)        
        
    #para secuencias periodicas
    elif type_target=='sec':
        amp=np.ones(N)*amp0
        phase=np.linspace(0,4*np.pi,N)
        omega=4*np.pi/itmax*np.ones(N)
        for it in range(itmax):
            target[:,it]=amp*np.cos(it*omega+phase)
    
    #generacion targe secuencias gaussianas
    elif type_target=='gauss':
        
        gg=np.zeros(N)
        sg=0.1*N            # ancho de la gaussiana. trelativo al tamanio del sistema
        omegagauss=0.5       # velocidad de desplazamiento
        for i in range(N):
            gg[i]=amp0*np.exp(-(i-N/2)**2/(2*sg**2))
        for it in range(itmax):
            target[:,it]=np.roll(gg,int(omegagauss*it))
            
    #OU process
    elif type_target=='ou':
        target = np.zeros(shape=(N,itmax))
        b_ou = 1/200
        mu = 0
        sigma = 0.5
       
        for it in range(itmax-1):
            target[:,it+1]= target[:,it]+b_ou*(mu - target[:,it])*dt + sigma*np.sqrt(dt)*np.random.randn(N)


            
    return target 


def guardar_matriz_csv(matriz, nombre_archivo):
    with open(nombre_archivo, 'w', newline='') as archivo_csv:
        escritor_csv = csv.writer(archivo_csv)
        for fila in matriz:
            fila_lista = [str(elemento) for elemento in fila.flat]
            escritor_csv.writerow(fila_lista)
            
def motifs(w,gsyn,N):
    
    w=w-np.mean(w)
    
    ww=np.matmul(w,w)
    wtw=np.matmul(w.T,w)
    wwt=np.matmul(w,w.T)
    
    sigma2=np.trace(wwt)/N
    
    tau_rec=np.trace(ww)
    tau_rec/=sigma2*N
    
    tau_div=np.sum(wwt)-np.trace(wwt)
    tau_div/=sigma2*N*(N-1)
    
    tau_con=np.sum(wtw)-np.trace(wtw)
    tau_con/=sigma2*N*(N-1)
    
    tau_chn=2*(np.sum(ww)-np.trace(ww))
    tau_chn/=sigma2*N*(N-1)
    
    return sigma2,tau_rec,tau_div,tau_con,tau_chn

def dynamics(x_var,r_var,I_var,nqif):
    dx=np.zeros(N)
    #LIF
    dx[nqif:] = -x_var[nqif:] + I_var[nqif:] + np.random.randn(N - nqif)*sigman 
    #QIF
    dx[:nqif] = 1 - np.cos(x_var[:nqif]) + I_var[:nqif]*(1 + np.cos(x_var[:nqif])) + np.random.randn(nqif)*sigman
       
    dr = -b*r_var
    return dx,dr


def detect(x,xnew,rnew,nspike,nqif):
     #LIF
     ispike_lif=np.where(x[nqif:]<vt) and np.where(xnew[nqif:]>vt)
     ispike_lif=ispike_lif[0]+nqif
     if(len(ispike_lif)>0):
         rnew[ispike_lif[:]] = rnew[ispike_lif[:]] + b
         xnew[ispike_lif[:]] = 0
         nspike[ispike_lif[:]] = nspike[ispike_lif[:]] + 1
     #QIF 
     dpi=np.mod(np.pi - np.mod(x,2*np.pi),2*np.pi)  # distancia a pi
     ispike_qif=np.where((xnew[:nqif]-x[:nqif])>0) and np.where((xnew[:nqif]-x[:nqif]-dpi[:nqif])>0)
     if(len(ispike_qif)>0):
         rnew[ispike_qif[:]] = rnew[ispike_qif[:]] + b
         nspike[ispike_qif[:]] = nspike[ispike_qif[:]] + 1
     return xnew,rnew,nspike


def plot_combined_graphs(w, modt, modw, cc, N2):
    # Grafico 1

    plt.figure(figsize=(14, 3))
    plt.subplot(1, 2, 1)
    plt.title("Weight matrix")
    plt.imshow(w)
    plt.colorbar()

    # Grafico 2
    plt.subplot(1, 2, 2)
    plt.xlabel('iteraciones')
    plt.ylabel('Pearson CC')
    plt.plot(cc, '-o')  
    plt.show() 

    plt.figure(figsize=(14, 10))
    # Grafico 3
    plt.subplot(3, 2, 1)
    plt.xlabel('time')
    plt.ylabel('Log10(|dw|/|w0|)')
    plt.ylim(-10, -2)
    plt.plot(modt, modw)

    # Grafico 4
    plt.subplot(3, 2, 2)
    plt.xlabel("w")
    plt.ylabel("N(w)")
    plt.yscale("log")
    plt.xlim(-2, 2)
    std = np.std(w)
    plt.text(1, 10000, "std=" + str(std)[:5], ha='right', va='top', bbox=dict(facecolor='white', alpha=0.5))
    counts, bins = np.histogram(w, bins=100)
    plt.hist(bins[:-1], bins, weights=counts)

    # Grafico 5
    plt.subplot(3, 2, 3)
    plt.xlim(-2, 2)
    plt.xlabel("w11")
    plt.ylabel("N(w11)")
    plt.yscale("log")
    std = np.std(w[:N2, :N2])
    plt.text(0.95, 0.95, f"std={std:.5f}", transform=plt.gca().transAxes, ha='right', va='top', bbox=dict(facecolor='white', alpha=0.5))
    counts, bins = np.histogram(w[:N2, :N2], bins=100)
    plt.hist(bins[:-1], bins, weights=counts)

    # Grafico 6
    plt.subplot(3, 2, 4)
    plt.xlim(-2, 2)
    plt.xlabel("w22")
    plt.ylabel("N(w22)")
    plt.yscale("log")
    std = np.std(w[N2:, N2:])
    plt.text(0.95, 0.95, f"std={std:.5f}", transform=plt.gca().transAxes, ha='right', va='top', bbox=dict(facecolor='white', alpha=0.5))
    counts, bins = np.histogram(w[N2:, N2:], bins=100)
    plt.hist(bins[:-1], bins, weights=counts)

    # Grafico 7
    plt.subplot(3, 2, 5)
    plt.xlim(-2, 2)
    plt.xlabel("w12")
    plt.ylabel("N(w12)")
    plt.yscale("log")
    std = np.std(w[:N2, N2:])
    plt.text(0.95, 0.95, f"std={std:.5f}", transform=plt.gca().transAxes, ha='right', va='top', bbox=dict(facecolor='white', alpha=0.5))
    counts, bins = np.histogram(w[:N2, N2:], bins=100)
    plt.hist(bins[:-1], bins, weights=counts)

    # Grafico 8
    plt.subplot(3, 2, 6)
    plt.xlim(-2, 2)
    plt.xlabel("w21")
    plt.ylabel("N(w21)")
    plt.yscale("log")
    std = np.std(w[N2:, :N2])
    plt.text(0.95, 0.95, f"std={std:.5f}", transform=plt.gca().transAxes, ha='right', va='top', bbox=dict(facecolor='white', alpha=0.5))
    counts, bins = np.histogram(w[N2:, :N2], bins=100)
    plt.hist(bins[:-1], bins, weights=counts)

    plt.tight_layout()
    plt.show()


def dpr_bias(ccorr,N,nloop):
    a=np.extract(np.identity(N),ccorr)
    c=np.extract(1-np.identity(N),ccorr)
    am2=np.mean(a)**2
    astd2=np.var(a)*N/(N-1)
    cm2=np.mean(c)**2
    cstd2=np.var(c)*N*(N-1)/(N*(N-1)-2)
    
    astd_bias2=astd2*(nloop-1)/(nloop+1) -2*(am2-cm2)/(nloop-1)+ 2*cstd2/(nloop+1)
    cstd_bias2=(nloop-1)*cstd2/nloop - (am2-cm2)/nloop -4*(cm2-np.sqrt(am2*cm2))/(nloop*(N+1))
    
    dpr_bias=N/(1+(astd_bias2/am2)+(N-1)*((cstd_bias2/am2)+(cm2/am2)))
    
    return dpr_bias


## SIMULACION

romega1_vec = np.array([1])
romega2_vec = np.array([5])
num_simulacion = 9

for i in range(len(romega1_vec)):
    print(i)
    romega1 = romega1_vec[i]
    romega2 = romega2_vec[i]
    
    num_simulacion +=  1

    directorios = crear_carpetas(num_simulacion)
    
    nombre_carpeta = directorios[0]
    nombre_subcarpeta_act = directorios[1]
    nombre_subcarpeta_pesos = directorios[2]
    
    target = generate_target(type_target, romega1 = romega1, romega2 = romega2, num_simulacion= num_simulacion, nombre_carpeta=nombre_carpeta)
    #file donde voy a guardar los resultados (CC, taus)
    filename_resultados = f'simulacion_{num_simulacion}_resultados.csv'
    csv_file_path = os.path.join(nombre_carpeta, filename_resultados)
    column_names = [ 'pqif' ,'seed','nloop', 'cc_lif', 'cc_qif', 'cc', 'sigma2','tau_rec','tau_div','tau_con','tau_chn']


    crear_archivo_parametros(filename_resultados, num_simulacion, nombre_carpeta)

    # Create or open the CSV file in append mode
    with open(csv_file_path, mode='a', newline='') as file:


        writer = csv.writer(file) 

        # Write column names to the CSV file if the file is empty (on the first iteration)
        if file.tell() == 0:
            writer.writerow(column_names)

        for pqif in [1]:
            
        
            nqif=int(N*pqif)

            
            for seed in range(cant_seed):
                print('Semilla:', seed)
                np.random.seed(seed = seed)
                itrain=np.random.binomial(1,ftrain,N)

                x=np.random.uniform(size=N)*2*np.pi            # variables internas neurona theta 
                r=np.zeros(N)                           # salida sinaptica generada por cada neurona
                nspike=np.zeros(N)
                rout=np.zeros((1,N))                               # salidas a graficar

                # corriente externa
                Iext=np.zeros((N,itmax))
                Ibac=amp_corriente*(2*np.random.uniform(size=N)-1)
                for it in range(itstim):                # estimulo externo duramte 10% del ciclo
                    Iext[:,it]=Ibac    
                    
                # definicion matriz de conectividad
                w= (sparse.random(N,N,p,data_rvs=np.random.randn)).todense()    # matriz de conexiones
                np.fill_diagonal(w,0)                                           # no autapses
                w*=gsyn/np.sqrt(p*N)                                            # normalizacion
                for i in range(N):                                              # suma de filas -> 0
                    i0=np.where(w[i,:])[1]
                    av0=0#     
                    if(len(i0)>0):    
                        av0=np.sum(w[i,i0])/len(i0)
                        w[i,i0]=w[i,i0]-av0                                       
                w0=np.copy(w)                                                   # guardo matriz inicial
                
                # matrices de correlacion de las entradas
                nind=np.zeros(N).astype('int')
                idx=[]
                P=[]
                for i in range(N):
                    ind=np.where(w[i,:])[1]
                    nind[i]=len(ind)
                    idx.append(ind)
                    P.append(np.identity(nind[i])/alpha)    
                
                
                # acumulacion modificacion de matrices
                modw=[]
                modt=[]


                for iloop in range(nloop): 
                    print('iloop: ',iloop) 

                    t=0
                    for it in range(itmax):
                        #RK2
                        II=np.squeeze(np.asarray(Iext[:,it]))
                        v=np.matmul(w,r.T)
                        v=np.squeeze(np.asarray(v))
                                        
                        dx,dr=dynamics(x,r,II+v,nqif)            
                        xnew=x+dt*dx/2
                        rnew=r+dt*dr/2
                        dx,dr=dynamics(xnew,rnew,II+v,nqif)

                        xnew=x+dt*dx
                        rnew=r+dt*dr           
                        
                        xnew,rnew,nspike = detect(x,xnew,rnew,nspike,nqif)
                        
                        t+=dt
                        x=np.copy(xnew)
                        r=np.copy(rnew)
                        rout=np.vstack([rout,r[iout]])

                        # aprendizaje
                        if  iloop>0  and iloop <= nloop_train and int(it>itstim):
                            error=target[:,it:it+1]-np.matmul(w,r.reshape(N,1))    
                            w1=np.zeros(w.shape)
                            for i in range(N):
                                ri=r[idx[i]].reshape(len(idx[i]),1)         # vector columna
                                k1=np.matmul(P[i],ri)
                                k2=np.matmul(ri.T,P[i])
                                den=1+np.matmul(ri.T,k1)[0]
                                
                                dP = -np.matmul(k1,k2)/den
                                P[i]+=dP                                    # correccion matriz
                                
                                dw = error[i,0]*np.matmul(P[i],r[idx[i]])*itrain[i]
                                w[i,idx[i]]+=dw                             # correccion pesos
                                w1[i,idx[i]]+=dw                            # acumulcion modificacion para norma
                            if it%10==0:
                                modt.append(np.array(it+iloop*itmax))
                                modw.append(np.array((np.log(np.linalg.norm(w1)/np.linalg.norm(w0)))))

                    sigma2,tau_rec,tau_div,tau_con,tau_chn=motifs(w,gsyn,N)

                    
                    Rmotifs=(1.0-tau_div-tau_con+tau_rec-2.0*tau_chn)/np.sqrt(1-tau_div-tau_con)
                    sigma=np.sqrt(sigma2)
                    R=Rmotifs*sigma

                    


                    if iloop == 0 or iloop == (nloop_train + 1):
                        path_w_seed = os.path.join(nombre_subcarpeta_pesos, f'simulacion_{num_simulacion}_pesos_pqif_{pqif}_matriz_iloop_{iloop}_semilla_{seed}')
                        guardar_matriz_csv(w, path_w_seed)
                        
                    # Pearson correlation
                    ci=0
                    ci_lif = 0
                    ci_qif = 0
                    for i in range(N):
                        m1=1+itstim+iloop*itmax
                        m2=m1+itmax-itstim
                        # Check if the input arrays have zero variance

                        if np.var(target[i, itstim:]) > 0 and np.var(rout[m1:m2, i]) > 0:
                            ci += pearsonr(target[i, itstim:], rout[m1:m2, i])[0] * itrain[i]
                            if(i < N/2):
                                ci_lif += pearsonr(target[i, itstim:], rout[m1:m2, i])[0] * itrain[i]
                            if (i >= N/2):
                                ci_qif += pearsonr(target[i, itstim:], rout[m1:m2, i])[0] * itrain[i]


                    ci_lif/=int(ftrain*N/2)
                    ci_qif/=int(ftrain*N/2)
                    ci /= int(ftrain*N)




                    writer.writerow([
                        pqif,
                        seed,
                        iloop,
                        ci_lif,
                        ci_qif,
                        ci,
                        sigma2,
                        tau_rec,
                        tau_div,
                        tau_con,
                        tau_chn, 
                            
                    
                    ])
                    

                # graficos
                def plots(iplot, path):
                    
                    fig = plt.figure(figsize=(16,4))

                    sub1 = fig.add_subplot(1,4,1)
                    sub1.title.set_text('Target')
                    sub1.set_xlim(itstim,itmax)
                    sub1.set_ylim(-amp0,amp0)
                    sub1.plot(target[iout[iplot],:])

                    sub2 = fig.add_subplot(1,4,2)
                    sub2.title.set_text(f'actividad pre-training, ipqif = {pqif}')
                    sub2.set_xlim(itstim,itmax)
                    sub2.plot(rout[:,iplot])

                    sub3 = fig.add_subplot(1,4,3)
                    sub3.title.set_text('actividad último loop de training')
                    sub3.set_xlim((nloop_train)*itmax+itstim,(nloop_train+1)*itmax)

                    sub3.plot(rout[:,iplot])
                    sub4 = fig.add_subplot(1,4,4)
                    sub4.title.set_text('actividad post-training')
                    sub4.set_xlim((nloop-1)*itmax+itstim,nloop*itmax)
                    sub4.plot(rout[:,iplot])


                    plt.savefig(path+str(iplot)+'.png')
                    plt.close(fig)
                    
                    #plt.plot(target[iout[iplot],:])
                    #plt.plot(rout[2500:4000, iplot])
                    #plt.show()

                nombre_subsub_act_pqif  = f"simulacion_{num_simulacion}_act_pqif_{pqif}"
                dir_act_pqif = crear_subcarpeta(nombre_subcarpeta_act, nombre_subsub_act_pqif)
                dir_act_pqif_seed = (os.path.join(dir_act_pqif , f"simulacion_{num_simulacion}_ej_actividad_semilla_{seed}"))

                if not os.path.exists(dir_act_pqif_seed):
                    os.makedirs(dir_act_pqif_seed)


                path = os.path.join(dir_act_pqif_seed, f"simulacion_{num_simulacion}_act_pqif_{pqif}_seed_{seed}_neurona_")

                for iplot in i_graph:
                    plots(iplot, path)