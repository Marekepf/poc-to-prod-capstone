from flask import Flask, request, jsonify, redirect, url_for
from run import TextPredictionModel

app = Flask(__name__)

model_artefacts_path = "C:/Users/Utilisateur/Desktop/DATA_S9/PocToProd/poc-to-prod-capstone/train/data/artefacts/2024-01-09-15-23-28"
model = TextPredictionModel.from_artefacts(model_artefacts_path)

@app.route('/')
def home():
    return '''
    <!DOCTYPE html>
    <html>
    <head>
    <title>Welcome to Marek's Web App for PocToProd</title>
    </head>
    <body>
        <h2>Welcome to Marek's Web App for PocToProd </h2>
        <form action="/predict">
            <input type="submit" value="Go to Prediction" />
        </form>
    </body>
    </html>
    '''

@app.route('/predict', methods=['POST', 'GET'])
def predict():
    if request.method == 'POST':
        text = request.form.get('text')
        predictions = model.predict([text])

        # Ensure each element in predictions is serializable
        serializable_predictions = [str(prediction) for prediction in predictions]

        return jsonify(predictions=serializable_predictions, text=text)

    return '''
    <!DOCTYPE html>
    <html>
    <head>
    <title>Prediction Text</title>
    </head>
    <body>
        <h2>Enter text for prediction</h2>
        <form method="post" action="/predict">
            <textarea name="text" rows="4" cols="50"></textarea>
            <br><br>
            <input type="submit" value="Predict">
        </form>
    </body>
    </html>
    '''

if __name__ == '__main__':
    app.run(debug=True)
