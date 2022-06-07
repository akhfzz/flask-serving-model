from server import ServeModelTF, api, app

api.add_resource(ServeModelTF, '/predict/')

if __name__ == "__main__":
    app.run(debug=True)
