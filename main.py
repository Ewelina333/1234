#Na 4 - API, które posiada 2 endpointy, jeden z punktu a., drugi GET który w parametrze
#otrzymuje link do zdjęcia, które jest w Internecie, pobiera je, a następnie zwraca
#informację o tym ile znaleziono osób na zdjęciu.

import cv2
import numpy as np
import urllib.request
from flask import Flask, request
from flask_restful import Resource, Api

# initialize the HOG descriptor/person detector
hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

app = Flask(__name__)
api = Api(app)


class PeopleCounterStatic(Resource):
    def get(self):
        # load image
        image = cv2.imread('ludzie.jpeg')
        image = cv2.resize(image, (700, 400))

        # detect people in the image
        (rects, weights) = hog.detectMultiScale(image, winStride=(4, 4), padding=(8, 8), scale=1.05)

        return {'peopleCount': len(rects)}


class PeopleCounterDynamicUrl(Resource):
    def get(self):
        # Get the URL from the request parameters
        url = request.args.get('url')

        try:
            # Download the image from the URL
            req = urllib.request.urlopen(url)
            arr = np.asarray(bytearray(req.read()), dtype=np.uint8)
            image = cv2.imdecode(arr, -1)

            # Resize the image if needed
            image = cv2.resize(image, (700, 400))

            # Detect people in the image
            (rects, weights) = hog.detectMultiScale(image, winStride=(4, 4), padding=(8, 8), scale=1.05)

            return {'peopleCount': len(rects)}

        except Exception as e:
            return {'error': str(e)}


api.add_resource(PeopleCounterStatic, '/')
api.add_resource(PeopleCounterDynamicUrl, '/dynamic')

if __name__ == '__main__':
    app.run(debug=True)
