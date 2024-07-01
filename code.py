import pandas as pd
import matplotlib.pyplot as plt

# Cargar datos desde el archivo CSV
data = pd.read_csv('datos_filtrados.csv', names=['Palabras Clave', 'Compartir'])

# Calcular la media de palabras clave para diferenciar en el gráfico
media_palabras = data['Palabras Clave'].mean()

# Separar datos para el gráfico
menos_media = data[data['Palabras Clave'] <= media_palabras]
mas_media = data[data['Palabras Clave'] > media_palabras]

# Graficar los datos
plt.figure(figsize=(10, 6))

# Pintar en azul los puntos con menos de media_palabras palabras
plt.scatter(menos_media['Palabras Clave'], menos_media['Compartir'], color='blue', label='Menos de media palabras')

# Pintar en naranja los puntos con más de media_palabras palabras
plt.scatter(mas_media['Palabras Clave'], mas_media['Compartir'], color='orange', label='Más de media palabras')

plt.xlabel('Palabras Clave')
plt.ylabel('Compartir')
plt.title('Relación entre Palabras Clave y Compartir de Artículos sobre ML')
plt.legend()
plt.grid(True)

# Guardar la gráfica como imagen PNG
plt.savefig('grafica_compartir_vs_palabras.png')

# Mostrar la gráfica
plt.show()
