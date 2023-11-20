from flask import Flask, request, jsonify
from flask_cors import CORS
import sqlite3
import datetime



print("starting ML training")
from main import *
print("ML execution completed\nstart backend service")

connection = sqlite3.connect("database.db")
cursor = connection.cursor()
cursor.execute('''CREATE TABLE IF NOT EXISTS logs( 
time timestamp, 
category INTEGER,
url VARCHAR(50));''')
connection.commit()
del connection
del cursor


app = Flask(__name__)
CORS(app)



@app.route('/get_stats', methods=['GET'])
def get_data():
    connection = sqlite3.connect("database.db")
    cursor = connection.cursor()
    cursor.execute("SELECT * FROM logs")
    data = cursor.fetchall()
    result = {
        "progress1": 0,
        "progress2": 0,
        "progress3": 0,
        "progress4": 0,
        "progress5": 0,
        "progress6": 0,
        "progress7": 0,
        "progress8": 0,
        "progress9": 0,
        "progress10": 0,
        "progress11": 0,
        "progress12": 0,
        "progress13": 0,
        "progress14": 0,
        "progress15": 0,
        "progress16": 0
    }
    for i in data:
        # print(i)
        if int(i[1]) == 1:
            result["progress1"] += 1
        elif int(i[1]) == 2:
            result["progress2"] += 1
        elif int(i[1]) == 3:
            result["progress3"] += 1
        elif int(i[1]) == 4:
            result["progress4"] += 1
        elif int(i[1]) == 5:
            result["progress5"] += 1
        elif int(i[1]) == 6:
            result["progress6"] += 1
        elif int(i[1]) == 7:
            result["progress7"] += 1
        elif int(i[1]) == 8:
            result["progress8"] += 1
        elif int(i[1]) == 9:
            result["progress9"] += 1
        elif int(i[1]) == 10:
            result["progress10"] += 1
        elif int(i[1]) == 11:
            result["progress11"] += 1
        elif int(i[1]) == 12:
            result["progress12"] += 1
        elif int(i[1]) == 13:
            result["progress13"] += 1
        elif int(i[1]) == 14:
            result["progress14"] += 1
        elif int(i[1]) == 15:
            result["progress15"] += 1
        elif int(i[1]) == 16:
            result["progress16"] += 1
        else:
            pass
            # print("error in log processing")
    return jsonify(result)


@app.route('/add_to_logs', methods=['POST'])
def api():
    website = str(request.json['url'])
    # website= input("enter website:")
    scrapTool = ScrapTool()
    connection = sqlite3.connect("database.db")
    cursor = connection.cursor()
    # try:
    web=dict(scrapTool.visit_url(website))
    text=(clean_text(web['website_text']))
    t=fitted_vectorizer.transform([text])
    # print(id_to_category[m1.predict(t)[0]])
    category = (id_to_category[m1.predict(t)[0]]).split()[-1]
    # print(type(category), category)
    if category == "Travel":
        query = "INSERT INTO logs values ('" + str(datetime.datetime.now()) + "', 1, '" + website + "')"
        cursor.execute(query)
    elif category == "Social Networking and Messaging":
        query = "INSERT INTO logs values ('" + str(datetime.datetime.now()) + "', 2, '" + website + "')"
        cursor.execute(query)
    elif category == "News":
        query = "INSERT INTO logs values ('" + str(datetime.datetime.now()) + "', 3, '" + website + "')"
        cursor.execute(query)
    elif category == "Streaming Services":
        query = "INSERT INTO logs values ('" + str(datetime.datetime.now()) + "', 4, '" + website + "')"
        cursor.execute(query)
    elif category == "Sports":
        query = "INSERT INTO logs values ('" + str(datetime.datetime.now()) + "', 5, '" + website + "')"
        cursor.execute(query)
    elif category == "Photography":
        query = "INSERT INTO logs values ('" + str(datetime.datetime.now()) + "', 6, '" + website + "')"
        cursor.execute(query)
    elif category == "Law and Government":
        query = "INSERT INTO logs values ('" + str(datetime.datetime.now()) + "', 7, '" + website + "')"
        cursor.execute(query)
    elif category == "Health and Fitness":
        query = "INSERT INTO logs values ('" + str(datetime.datetime.now()) + "', 8, '" + website + "')"
        cursor.execute(query)
    elif category == "Games":
        query = "INSERT INTO logs values ('" + str(datetime.datetime.now()) + "', 9, '" + website + "')"
        cursor.execute(query)
    elif category == "E-Commerce":
        query = "INSERT INTO logs values ('" + str(datetime.datetime.now()) + "', 10, '" + website + "')"
        cursor.execute(query)
    elif category == "Forums":
        query = "INSERT INTO logs values ('" + str(datetime.datetime.now()) + "', 11, '" + website + "')"
        cursor.execute(query)
    elif category == "Food":
        query = "INSERT INTO logs values ('" + str(datetime.datetime.now()) + "', 12, '" + website + "')"
        cursor.execute(query)
    elif category == "Education":
        query = "INSERT INTO logs values ('" + str(datetime.datetime.now()) + "', 13, '" + website + "')"
        cursor.execute(query)
    elif category == "Computers and Technology":
        query = "INSERT INTO logs values ('" + str(datetime.datetime.now()) + "', 14, '" + website + "')"
        cursor.execute(query)
    elif category == "Business/Corporate":
        query = "INSERT INTO logs values ('" + str(datetime.datetime.now()) + "', 15, '" + website + "')"
        cursor.execute(query)
    elif category == "Adult":
        query = "INSERT INTO logs values ('" + str(datetime.datetime.now()) + "', 16, '" + website + "')"
        cursor.execute(query)
    else:
        print("error: CATEGORY NOT IDENTIFIED")
        return jsonify({"response": "0"})
    connection.commit()
    return jsonify({"response": "1"})
        # data=pd.DataFrame(m1.predict_proba(t)*100,columns=df['Category'].unique())
        # data=data.T
        # data.columns=['Probability']
        # data.index.name='Category'
        # a=data.sort_values(['Probability'],ascending=False)
        # a['Probability']=a['Probability'].apply(lambda x:round(x,2))
    # except:
        # print("Connection Timedout!")
        # return jsonify({"response": "Connection Timedout!"})

# if __name__==__main__:
# app.run(host="0.0.0.0", port=5000, debug=True)
app.run(port=5000)