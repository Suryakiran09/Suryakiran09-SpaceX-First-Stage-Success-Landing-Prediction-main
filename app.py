from flask import Flask , render_template , request
app = Flask(__name__) 
import pickle
model = pickle.load(open('model.pkl' , 'rb'))

@app.route('/')
def helloworld():
    return render_template('ind.html')

@app.route('/login' , methods = ['POST'])
def login():
    py = request.form["py"]
    orb = request.form["or"]
    
    ou = request.form["ou"]
    g = request.form["g"]
    r = request.form["r"]
    l = request.form["l"]
    L = request.form["L"]
    B = request.form["B"]
    R = request.form["R"]
    se = request.form["se"]
    lo = request.form["lo"]
    la = request.form["la"]
    t = [[int(py),int(orb),int(ou),int(g),int(r),int(l),int(L),int(B),int(R),int(se), float(lo), float(la)]]
    output= model.predict(t)
    print(output)
    if output[0] == 0:
        return render_template('ind.html' , Y = "It is Unsuccessfull")
    elif output[0] == 1:
        return render_template('ind.html', Y = "It is Successfull") 
if __name__ == '__main__' :
    app.run(debug= True)
    
        
    