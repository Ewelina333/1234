import cv2
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
    @property
    def get(self):
        # TODO:
        # 1. Pobrać zdjęcie z otrzymanego adresu
        # 2. Pobrane zdjęcie można zapisać na dysku lub przetwarzać je w pamięci podręcznej
        # 3. Załadowane zjęcie do zmiennej image przekazać do algorytmu hog.detectMultiScale i zwrócić z
        # endpointu liczbę wykrytych osób.

        url = request.args.get('url')
        print('url', url)
        return {'peopleCount': 0}

api.add_resource(PeopleCounterStatic, '/')
api.add_resource(PeopleCounterDynamicUrl, '/dynamic')

if __name__ == '__main__':
    app.run(debug=True)
