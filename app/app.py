from server import ServeModel, api, app

api.add_resource(ServeModel, '/predict/')

if __name__ == "__main__":
    app.run(debug=True)
