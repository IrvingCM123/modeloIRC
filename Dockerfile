# Usa una imagen base que tenga Python instalado
FROM python:3.10

# Establece el directorio de trabajo
WORKDIR /app

# Copia los archivos necesarios (asegúrate de tener los archivos necesarios en el contexto de construcción)
COPY . .

# Instala las dependencias de tu aplicación Python
RUN pip install -r requirements.txt  # Ajusta el nombre de tu archivo de requerimientos si es diferente

# Expone el puerto en el que tu API estará escuchando
EXPOSE 5000

# Comando para ejecutar tu aplicación Python
CMD ["python", "catdog_client.py"]  # Ajusta el nombre de tu archivo de aplicación si es diferente
