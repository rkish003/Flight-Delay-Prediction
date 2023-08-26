from flask import Flask,request, url_for, redirect, render_template
import pickle
import numpy as np
from sklearn.preprocessing import LabelEncoder
from flight_delay import mapping_dict,dep_air_list,airline_list,arr_air_list
application = Flask(__name__,template_folder='templates')

model=pickle.load(open('model.pkl','rb'))

@application.route('/')
def hello_world():
    return render_template("flight_delay.html",airports=dep_air_list,airline=airline_list)

@application.route('/visualization')
def visualization_page():
    return render_template('visualization.html')


@application.route('/predict',methods=['POST','GET'])
def predict():
    int_features=[str(x) for x in request.form.values()]
    print(int_features)
    in_clms=['DEP_AIRPORT','ARR_AIRPORT','AIRLINE']
    final=[]
    for i in range(len(in_clms)):
        final.append(mapping_dict[in_clms[i]][int_features[i]])
    final=[np.array(final)]
    print(int_features)
    print(final)
    prediction=model.predict(final)
    print(prediction)
    output = [i for i in mapping_dict['ARR_STATUS'] if mapping_dict['ARR_STATUS'][i]==prediction[0]]
    return render_template("flight_delay.html",pred=output[0],airports=dep_air_list,airline=airline_list)
    # output='{0:.{1}f}'.format(prediction[0][1], 2)

    # if output==str(-1):
    #     return render_template("flight_delay.html",pred='Your Forest is in Danger.\nProbability of fire occuring is {}'.format(output),bhai="kuch karna hain iska ab?")
    # else:
    #     return render_template("flight_delay.html",pred='Your Forest is safe.\n Probability of fire occuring is {}'.format(output),bhai="Your Forest is Safe for now")


if __name__ == '__main__':
    application.run(debug=True)
