from flask import Flask, request
from flask_restful import Resource, Api
from model import Model
app = Flask(__name__)
api = Api(app)

class TradePredict(Resource):
    def get(self):
        '''
        Import the model
            write prediction module
                outputs datetime, price, trade amount as JSON
        pull data from poloniex
        use model.prediction
        return model.prediction output
        '''
        m = model(symbols = ['BTC'])

        return model.predict_for_api(num = 5280)

api.add_resource(TradePredict, '/')

if __name__ == '__main__':
    app.run(debug=True)
