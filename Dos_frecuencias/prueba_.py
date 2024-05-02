import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os


def csv_to_matrix(filename):
    df = pd.read_csv(filename, header=None) 
    matrix = df.values 
    return matrix

def cargar_archivo_parametros(filename_parametros):
    # Lee el archivo CSV
    df = pd.read_csv(filename_parametros)
    
    # Convierte el DataFrame a un diccionario de Python
    parametros = df.to_dict(orient='records')[0]
    
    # Retorna el diccionario de parámetros
    return parametros


#Defino los contenidos de los distintos archivos

#Prueba6: targets con una sola frecuencia (romega = 1) y desfasaje presente entre neuronas
num_sim = 0
nombre_archivo = nombre_archivo = f'/home/martina/Tesis_2024/Dos_frecuencias/simulacion_{num_sim}/simulacion_{num_sim}_resultados.csv'
data = pd.read_csv(nombre_archivo)



# Llama a la función cargar_archivo_parametros con el nombre del archivo CSV
nombre_carpeta = f"/home/martina/Tesis_2024/Dos_frecuencias/simulacion_{num_sim}"  # Reemplaza 'ruta/a/la/carpeta' con la ruta real de tu carpeta
filename_parametros = os.path.join(nombre_carpeta, f'simulacion_{num_sim}_parametros.csv')  # Reemplaza 'simulacion_1_parametros.csv' con el nombre real de tu archivo CSV
parametros = cargar_archivo_parametros(filename_parametros)

# Ahora puedes acceder a los parámetros como lo harías normalmente
N = parametros['N']
nloop = parametros['nloop']
cant_seed = parametros['cant_seed']

# Obtener los valores únicos de 'pqif'
pqif_values = data['pqif'].unique()

#quiero usar los mismos colores para los distintos valores de pqif

pqif_vector = [0, 0.5, 1]
colores = ['r', 'g', 'b']# %%


# Iterar sobre los valores únicos de 'pqif' para graficar todas las líneas en el subplot actual
for i in range(len(pqif_vector)):
    pqif_value = pqif_vector[i]
    color = colores[i]
    # Filtrar los datos para el valor actual de 'pqif'
    data_pqif = data[(data['pqif'] == pqif_value) & (data['cc'] > 0.5)]

    # Calcular el promedio y la desviación estándar para cada combinación de 'nloop' y 'columna'
    grouped_data = data_pqif.groupby(['nloop'])
    for columna in ['cc']:
        grouped_column_data = grouped_data[columna].agg(['mean', 'std']).reset_index()

        # Graficar cc vs nloop para cada 'pqif' con barras de error
        plt.errorbar(grouped_column_data['nloop'], grouped_column_data['mean'], yerr=grouped_column_data['std'], label=f'pqif={pqif_value}', color = color, fmt='-o')

    # Agregar leyenda al subplot actual
    plt.legend()

# Ajustar el diseño de los subplots
plt.tight_layout()
plt.grid()
plt.xlabel('nloop')
plt.ylabel('CC')

# Mostrar los subplots
plt.show()

df_nloop = data[(data['nloop'] == (nloop-1))]
min_val = np.min(df_nloop['tau_rec'])
max_val = np.max(df_nloop['tau_rec'])
    
for pq in range(len(pqif_vector)):

    pqif = pqif_vector[pq]
    color = colores[pq]
    df_nloop_pqif = df_nloop[(df_nloop['pqif'] == pqif)]
    bins = np.linspace(min_val, max_val, 12)
    plt.hist(df_nloop_pqif['tau_rec'], bins, edgecolor='black', label=f'pqif = {pqif}', color = color, alpha=0.5)
    plt.legend()
    plt.grid()
    plt.title('tau_rec')

plt.show()

cc_min = [0.3, 0.5, 0.7, 0.8]
min_val = np.min(df_nloop['tau_rec'])
max_val = np.max(df_nloop['tau_rec'])

for j in range(len(pqif_vector)):
    pqif = pqif_vector[j]
    df_nloop_16_pqif = df_nloop[(df_nloop['pqif'] == pqif)]

    for i in range(len(cc_min)):
        
        df_cc = df_nloop_16_pqif[df_nloop_16_pqif['cc']> cc_min[i]]
        bins = np.linspace(min_val, max_val, 12)
        plt.hist(df_cc['tau_rec'], bins, edgecolor='black', label=f'cc > {cc_min[i]},', alpha=0.5)
        plt.legend()
        plt.grid()
        plt.title(f'tau_rec, pqif = {pqif}')

    plt.show()
    


# Obtener los valores únicos de 'pqif'
pqif_values = df_nloop['pqif'].unique()

# Iterar sobre los valores únicos de 'pqif' para graficar todas las líneas en el subplot actual
for i in range(len(pqif_values)):
    pqif_value = pqif_values[i]
    color = colores[i]
    plt.scatter(df_nloop[df_nloop['pqif'] == pqif_value]['cc'], df_nloop[df_nloop['pqif'] == pqif_value]['tau_rec'], label=f'pqif={pqif_value}', color=color)
    plt.axhline(np.mean(df_nloop[df_nloop['pqif'] == pqif_value]['tau_rec']), color = color)

# Agregar leyenda al subplot actual
plt.legend()

# Ajustar el diseño de los subplots
plt.tight_layout()
plt.grid()
plt.xlabel('CC')
plt.ylabel('tau_rec')

# Mostrar los subplots
plt.show()


# Iterar sobre los valores únicos de 'pqif'
for pqif_value in df_nloop['pqif'].unique():
    # Filtrar los datos para el valor actual de 'pqif' y para aquellos donde 'cc' es mayor que 0.6
    filtered_data = df_nloop[(df_nloop['pqif'] == pqif_value) & (df_nloop['cc'] > 0.72)]
    
    # Obtener los valores únicos de 'seed' que cumplen con el criterio
    seed_values = filtered_data['seed'].unique()
    print(seed_values)

for i in range(len(pqif_vector)):
    pqif = pqif_vector[i]
    filename = f"/home/martina/Tesis_2024/Dos_frecuencias/simulacion_{num_sim}/simulacion_{num_sim}_matrices_pesos/simulacion_{num_sim}_pesos_pqif_{pqif}_matriz_iloop_11_semilla_0"
    weights_matrix = csv_to_matrix(filename)
    plt.imshow(weights_matrix, cmap='viridis', interpolation='nearest')

    # Añadir barra de color para representar los valores de los pesos
    plt.colorbar()

    # Añadir título y etiquetas de los ejes
    plt.title('Matriz de Pesos')
    plt.xlabel('Neurona de Entrada')
    plt.ylabel('Neurona de Salida')

    # Mostrar la gráfica
    plt.show()


