from flask import Flask, render_template, request, redirect, url_for
from model import evaluate_model, global_form_data

app = Flask(__name__)


# Define the route for the homepage that displays the form
@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        # Capture the form data and store it in the global variable
        global_form_data['name'] = request.form.get("name")
        global_form_data['email'] = request.form.get("email")
        global_form_data['symptoms'] = request.form.get("symptoms")

        # Send data for evaluation
        result = evaluate_model(global_form_data['symptoms'])

        # Display the result
        return f"Thank you, {global_form_data['name']}!<br>{result}"

    # HTML form
    return '''
        <h1>User Information Form</h1>
        <form method="post" action="/">
            <label for="name">Name:</label><br>
            <input type="text" id="name" name="name" required><br><br>

            <label for="email">Email:</label><br>
            <input type="email" id="email" name="email" required><br><br>

            <label for="symptoms">Describe your symptoms:</label><br>
            <textarea id="symptoms" name="symptoms" required></textarea><br><br>

            <input type="submit" value="Submit">
        </form>
    '''


if __name__ == "__main__":
    app.run(debug=True)
