from flask import Flask, request
from flask_restful import Resource, Api
from model import Model
app = Flask(__name__)
api = Api(app)

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
        m = Model(symbols = ['BTC'])

        return m.predict_for_api(5900, self.funds)

api.add_resource(TradePredict, '/')

if __name__ == '__main__':
    app.run(debug=True)
