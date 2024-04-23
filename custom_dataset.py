import os
import shutil

# Definir diretórios
data_dir = "./data/"
custom_dir = "./data/custom/"

# Criar diretório personalizado se não existir
os.makedirs(custom_dir, exist_ok=True)

# Percorrer as pastas de teste e validação
for folder in ["teste", "validacao"]:
    folder_path = os.path.join(data_dir, folder)
    # Percorrer as pastas de classes
    for class_folder in os.listdir(folder_path):
        class_folder_path = os.path.join(folder_path, class_folder)
        # Percorrer as pastas de recortes
        for recorte_folder in os.listdir(class_folder_path):
            recorte_folder_path = os.path.join(class_folder_path, recorte_folder)
            # Percorrer as pastas de segmento
            for segmento_folder in os.listdir(recorte_folder_path):
                segmento_folder_path = os.path.join(
                    recorte_folder_path, segmento_folder
                )
                # Percorrer as imagens
                for image in os.listdir(segmento_folder_path):
                    # Verificar se o arquivo é uma imagem
                    if image.lower().endswith((".png", ".jpg", ".jpeg")):
                        # Definir o caminho de origem e destino da imagem
                        src = os.path.join(segmento_folder_path, image)
                        dst = os.path.join(custom_dir, class_folder, image)
                        # Criar a pasta de destino se não existir
                        os.makedirs(os.path.dirname(dst), exist_ok=True)
                        # Copiar a imagem
                        shutil.copy(src, dst)
