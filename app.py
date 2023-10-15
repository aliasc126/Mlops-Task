from flask import Flask, render_template, request
import pickle

app = Flask(__name__ , template_folder='template')

@app.route('/')
def home():
    # load the home page
    return render_template('home.html')

@app.route('/result', methods=['POST'])
def result():
    output = request.form.to_dict()
    years= output['years']
    #load model.pkl
    
    model = pickle.load(open('model.pkl','rb'))
    #predict the result
    result = model.predict([[float(years)]])
    result = result[0]
    return render_template('home.html', result=result)

@app.route('/resultNotRender', methods=['POST'])
def resultNotRender():
    output = request.form.to_dict()
    years= output['years']
    #load model.pkl
    
    model = pickle.load(open('model.pkl','rb'))
    #predict the result
    result = model.predict([[float(years)]])
    return result[0]


if __name__ == '__main__':
    app.run(debug=True,host='0.0.0.0')