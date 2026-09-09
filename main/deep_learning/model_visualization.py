import visualkeras
from tensorflow.keras.models import load_model

model = load_model('./dave2_original.h5', compile=False)

# Gerar visualização
visualkeras.layered_view(model, 
                         legend=True, 
                         to_file='model_deeppicar.png',
                         scale_xy=15,  # Ajuste conforme necessário
                         spacing=50)   # Ajuste o espaçamento