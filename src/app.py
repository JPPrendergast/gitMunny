from flask import Flask, request
from flask_restful import Resource, Api
from model import Model
import keras.backend as K
from keras.models import load_model
app = Flask(__name__)
api = Api(app)

global m


class TradePredict(Resource):
    def put(self):
        self.funds = request.form['funds']
        # self.get()
        return self.get()

    def get(self):
        '''
        Import the model
            write prediction module
                outputs datetime, price, trade amount as JSON
        pull data from poloniex
        use model.prediction
        return model.prediction output
        '''
        m = Model(symbols=['BTC'])
        m.rnn = load_model('./data/model{}sgd.h5'.format(5900))
        response = m.predict_for_api(self.funds)
        K.clear_session()
        return response


api.add_resource(TradePredict, '/')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=80, debug=True, threaded=True)
