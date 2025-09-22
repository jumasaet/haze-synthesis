import os
import shutil
from pathlib import Path

def organizar_archivos(directorio_principal):
    # Crear los subdirectorios si no existen
    directorio_fog = Path(directorio_principal) / "fog"
    directorio_mask = Path(directorio_principal) / "mask"
    
    directorio_fog.mkdir(exist_ok=True)
    directorio_mask.mkdir(exist_ok=True)
    
    # Listar todos los archivos en el directorio
    for archivo in Path(directorio_principal).iterdir():
        if archivo.is_file():
            nombre_archivo = archivo.name
            
            # Procesar archivos _fog
            if nombre_archivo.endswith("_fog.png"):
                # Extraer solo el número (primeros 5 dígitos)
                numero = nombre_archivo.split('_')[0]
                nuevo_nombre = f"{numero}.png"
                destino = directorio_fog / nuevo_nombre
                shutil.move(str(archivo), str(destino))
                print(f"Movido: {nombre_archivo} -> fog/{nuevo_nombre}")
            
            # Procesar archivos _mask_cont
            elif nombre_archivo.endswith("_mask_cont.png"):
                # Extraer solo el número (primeros 5 dígitos)
                numero = nombre_archivo.split('_')[0]
                nuevo_nombre = f"{numero}.png"
                destino = directorio_mask / nuevo_nombre
                shutil.move(str(archivo), str(destino))
                print(f"Movido: {nombre_archivo} -> mask/{nuevo_nombre}")

if __name__ == "__main__":
    # Reemplaza con la ruta de tu directorio
    directorio = "M3FD/day_fog/fog/"  # Directorio actual, o especifica la ruta completa
    
    print("Organizando archivos...")
    organizar_archivos(directorio)
    print("¡Proceso completado!")