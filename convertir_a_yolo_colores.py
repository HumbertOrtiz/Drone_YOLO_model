import os
import json
import shutil
import random

# --- CONFIGURACIÓN ---
source_folders = [
    "json_gates1", 
    "json_gates4", 
    "json_gates5", 
    "json_gates6", 
    "json_gates7", 
    "json_gates8"
]

# Mapa de clases (ID 0=Rojo, 1=Verde, 2=Azul)
class_map = {
    "Red_gates": 0,
    "Green_gates": 1,
    "Blue_gates": 2,
    "Greeen_gates": 1  # Por si hay error de dedo en algún json viejo
}

output_dir = "Dataset_Final_Colores"
train_ratio = 0.8  # 80% entrenamiento, 20% validación

# ----------------------

def convert_to_yolo(size, box):
    # Convierte coordenadas de píxeles a formato normalizado YOLO (0 a 1)
    dw = 1. / size[0]
    dh = 1. / size[1]
    x = (box[0] + box[1]) / 2.0
    y = (box[2] + box[3]) / 2.0
    w = box[1] - box[0]
    h = box[3] - box[2]
    return (x * dw, y * dh, w * dw, h * dh)

def find_image(base_name, folder):
    # Busca la imagen jpg, png o jpeg en la misma carpeta que el json
    extensions = [".jpg", ".png", ".jpeg", ".JPG", ".PNG"]
    
    for ext in extensions:
        full_path = os.path.join(folder, base_name + ext)
        if os.path.exists(full_path):
            return full_path # ¡La encontramos!
            
    return None

def process_folders():
    # Limpiar carpeta de salida si ya existe para empezar desde cero
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    
    # Crear carpetas nuevas
    for split in ['train', 'val']:
        os.makedirs(os.path.join(output_dir, 'images', split), exist_ok=True)
        os.makedirs(os.path.join(output_dir, 'labels', split), exist_ok=True)

    print("Iniciando conversión inteligente...")
    total_images = 0
    missing_images = 0

    for folder in source_folders:
        print(f"--> Procesando carpeta: {folder}")
        if not os.path.exists(folder):
            print(f"    ADVERTENCIA: No existe la carpeta {folder}")
            continue

        # Listar todos los archivos JSON
        json_files = [f for f in os.listdir(folder) if f.endswith('.json')]
        
        for json_file in json_files:
            json_path = os.path.join(folder, json_file)
            base_name = os.path.splitext(json_file)[0]
            
            # 1. BUSCAR LA IMAGEN (Esta es la parte clave)
            image_path = find_image(base_name, folder)
            
            if not image_path:
                missing_images += 1
                # Si quieres ver cuáles faltan, descomenta la siguiente línea:
                # print(f"Falta imagen para: {json_file}")
                continue

            # 2. PROCESAR JSON
            with open(json_path, 'r') as f:
                try:
                    data = json.load(f)
                except:
                    print(f"    Error leyendo {json_file}")
                    continue
            
            im_w = data['imageWidth']
            im_h = data['imageHeight']
            yolo_lines = []
            
            found_valid_label = False
            for shape in data['shapes']:
                label = shape['label']
                
                if label in class_map:
                    class_id = class_map[label]
                    
                    points = shape['points']
                    x_coords = [p[0] for p in points]
                    y_coords = [p[1] for p in points]
                    
                    # Crear caja (xmin, xmax, ymin, ymax)
                    box = (min(x_coords), max(x_coords), min(y_coords), max(y_coords))
                    
                    # Convertir matemáticas
                    bb = convert_to_yolo((im_w, im_h), box)
                    
                    # Guardar línea: ID x y w h
                    yolo_lines.append(f"{class_id} {bb[0]} {bb[1]} {bb[2]} {bb[3]}")
                    found_valid_label = True
            
            # Solo guardamos si el archivo tiene etiquetas válidas (colores)
            if found_valid_label:
                # Decidir aleatoriamente si es train o val
                split = 'train' if random.random() < train_ratio else 'val'
                
                # Copiar imagen encontrada a la nueva carpeta
                img_ext = os.path.splitext(image_path)[1]
                target_img = os.path.join(output_dir, 'images', split, base_name + img_ext)
                shutil.copy(image_path, target_img)
                
                # Guardar archivo .txt con las coordenadas
                target_txt = os.path.join(output_dir, 'labels', split, base_name + ".txt")
                with open(target_txt, 'w') as out_f:
                    out_f.write('\n'.join(yolo_lines))
                
                total_images += 1

    print("-" * 30)
    print(f"¡Proceso terminado!")
    print(f"Imágenes procesadas correctamente: {total_images}")
    print(f"Imágenes perdidas: {missing_images}")
    print(f"Tu nuevo dataset está en: {output_dir}")

if __name__ == '__main__':
    process_folders()