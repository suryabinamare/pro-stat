from flask import Flask, render_template




from routes.home import home_bp
from routes.statvalues import statistics_bp
from routes.contact import contact_bp
from routes.distributions import distributions_bp
from routes.populationmean import populationmean_bp
from routes.populationproportion import populationproportion_bp
from routes.chisquare import chisquare_bp
from routes.linear import linear_bp
from routes.anova import anova_bp





app = Flask(__name__)
app.secret_key = "suryathapamagar08"

# Config here
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 10 * 1024 * 1024  # 10MB limit
app.config['ALLOWED_EXTENSIONS'] = {'csv', 'xlsx'}


# Register all blueprints
app.register_blueprint(home_bp)
app.register_blueprint(statistics_bp, url_prefix="/statvalues")
app.register_blueprint(contact_bp, url_prefix="/contact")
app.register_blueprint(distributions_bp, url_prefix="/distributions")
app.register_blueprint(populationmean_bp, url_prefix="/populationmean")
app.register_blueprint(populationproportion_bp, url_prefix="/populationproportion")
app.register_blueprint(chisquare_bp, url_prefix="/chisquare")
app.register_blueprint(linear_bp, url_prefix="/linear")
app.register_blueprint(anova_bp, url_prefix = "/anova")




# Ensure upload folder exists
import os
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)


@app.route("/")
def home():
    return render_template("sidebar.html", title="Home")



if __name__ == "__main__":
    app.run(debug=True)
