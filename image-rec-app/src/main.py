 #!/usr/bin/env python3

import flask
import os
import requests

app = flask.Flask(__name__)

predict_api_url = os.getenv('PREDICT_API_URL', 'http://localhost:8081/image')

def log(e):
    print("{0}\n".format(e))

@app.route('/', methods=['GET'])
def index():
    return flask.render_template("mnist.html")

@app.route('/image', methods=['POST'])
def image():
    log("PREDICT_API_URL=" + str(predict_api_url))

    # get the image bytes from HTTP request
    data=flask.request.data
    # Make the POST request to the server with the image bytes
    response = requests.post(predict_api_url, data=data)

    # Check if the request was successful
    if response.status_code == 200:
        return flask.Response(response.content, mimetype='application/json', status=200)
    else:
        print("Failed to get response from server, status code:", response.status_code)
        # Return an error response
        return flask.Response("Error calling the image processing API", status=response.status_code)
 
# Get the PORT from environment
port = os.getenv('PORT', '8080')
debug = os.getenv('DEBUG', 'false')
if __name__ == "__main__":
    log("application ready - Debug is " + str(debug))
    app.run(host='0.0.0.0', port=int(port), debug=debug)
