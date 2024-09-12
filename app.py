from flask import Flask, request, jsonify
import joblib
import numpy as np
import warnings

# Suppress warnings
warnings.filterwarnings("ignore")

app = Flask(__name__)

# Load the model
model = joblib.load("Depression_AI.pkl")

@app.route('/api/depression', methods=['POST'])
def depression():
    sadness = int(request.form.get('sadness')) 
    euphoric = int(request.form.get('euphoric')) 
    exhausted = int(request.form.get('exhausted'))
    sleep = int(request.form.get('sleep'))
    swing = int(request.form.get('swing'))
    Suicidal = int(request.form.get('suicidal'))
    anorxia = int(request.form.get('anorxia'))
    authority = int(request.form.get('authority'))
    explanation = int(request.form.get('explanation'))
    aggressive = int(request.form.get('aggressive'))
    move_on = int(request.form.get('move_on'))
    break_down = int(request.form.get('break_down'))
    admit = int(request.form.get('admit'))
    overthinking = int(request.form.get('overthinking'))
    sexual = int(request.form.get('sexual'))
    concentration = int(request.form.get('concentration'))
    optimisim = int(request.form.get('optimisim'))
    
    # Prepare the input for the model
    x = np.array([[sadness, euphoric, exhausted, sleep, swing, Suicidal, anorxia, authority,
     explanation, aggressive, move_on, break_down, admit, overthinking, sexual, concentration, optimisim]])

    # Predict using the model
    prediction = model.predict(x)

    if(prediction[0] == 0):
        result = "Normal"
    elif(prediction[0] == 1):
        result = "Bipolar Type-1"
    elif(prediction[0] == 2):
        result = "Bipolar Type-2"
    elif(prediction[0] == 3):
        result = "Depression"
    else:
        result = "Error!!"
    
    # Return the result
    return jsonify({'result': result, 'message': 'ประมวลผลเสร็จสิ้น', 'status': True}), 200

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=3000)