from flask import Flask, reder_template, request, jsonify
import tensorflow as tf
from PIL import Image, ImageOps
import numpy as np

app = Flask(__name__)

print(" Cargando el modelo de IA ")
modelo = tf.keras.models.load_model('modelo.h5')
print(" Modelo cargado correctamente ")
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predecir', methods=['POST'])
def predecir():
    if 'image' not in request.files:
        return jsonify({'error': 'No se ha proporcionado ningún imagen'}), 400

    archivo = request.files['image']
    if archivo.filename == '':
        return jsonify({'error': 'No se ha guardado ningún imagen'}), 400
    try:
            img = Image.open(archivo.stream).convert('L')
            img_invertida = ImageOps.invert(img)
            img_redimensionada = img_invertida.resize((28, 28))

            img_array = np.array(img_redimensionada) /255.0
            img_lista = np.expand_dims(img_array, axis=0)

            predicciones = modelo.predict(img_lista)
            numero_predicho = int(np.argmax(predicciones[0]))
            confianza = float(np.argmax(predicciones[0])*100)
            return jsonify({
            'numero_predicho': numero_predicho,
            'confianza': round(confianza, 2)
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)