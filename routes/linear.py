from flask import Flask, request, Blueprint, render_template, current_app

import base64
import io
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import sympy as sp
from sympy import latex, Eq, symbols
import seaborn as sns
from scipy.stats import f
from scipy.stats import norm
from scipy.stats import t
from scipy.stats import chi2
import scipy.stats as stats
from werkzeug.utils import secure_filename
from sklearn.linear_model import LinearRegression
from flask import Blueprint



linear_bp = Blueprint("linear", __name__)

@linear_bp.route("lineardescriptive", methods = ["GET", "POST"])
def descriptive():
    equation = None
    plot_url = None
    values = None

    if request.method == "POST":
        # Get form data
        x_values = request.form.getlist("x[]")
        y_values = request.form.getlist("y[]")
        file = request.files.get("file")

        if file and file.filename != "":
            if file.filename.endswith(".csv"):
                df = pd.read_csv(file)
            else:
                df = pd.read_excel(file)
            x1, y1 = df.iloc[:,0], df.iloc[:,1]
        else:
            if len(x_values) < 2 or len(y_values) < 2:
                equation = "❌ Please enter at least two points."
                return render_template("linear_descriptive.html", equation=equation, plot_url=plot_url, title="Linear Regression")
            else:
                # Convert to numpy
                x1 = np.array(x_values, dtype=float)
                y1 = np.array(y_values, dtype=float)
        n = len(y1)
        x_bar = np.mean(x1)
        y_bar = np.mean(y1)
        Sxx= np.sum(x1**2) - (np.sum(x1))**2/n
        Syy = np.sum(y1**2) - (np.sum(y1))**2/n
        Sxy = np.sum(x1*y1)-(np.sum(x1)*np.sum(y1))/n
        m = round(Sxy/Sxx,4)
        b = round(y_bar - m*x_bar, 4)
        R_squared = Sxy**2/(Sxx*Syy)
        sig = np.sqrt(R_squared)

        y_pred = m*x1 + b

        x,  y = sp.symbols('x y')
        eq = m*x + b
        eq1 = Eq(y, eq)
        equation = latex(eq1)
        se = np.sqrt(np.sum((y1 - y_pred)**2)/(n-2))
        SSE = np.sum((y1 - y_pred)**2)

        

        # Check which button was clicked
        action = request.form.get("action")
        if action == "plot" or action == "plot1":
            # Create scatter plot + regression line
            fig, axes = plt.subplots(1,3, figsize = (12,4))
            axes[0].scatter(x1, y1, color="blue", label="Data points")
            axes[0].plot(x1, y_pred, color="red", label="Regression line")
            for i in range(n):
                axes[0].plot([x1[i], x1[i]], [y1[i], y_pred[i]], color="gray", linestyle="--", linewidth=0.7)
            axes[0].scatter(x1,y_pred, marker = '*', s = 100, color = "green", label = "Predicted points")
            axes[0].set_title("Scatter Plot with Regression Line")
            axes[0].set_xlabel("X")
            axes[0].set_ylabel("Y or $\hat{Y}$")
            axes[0].legend()

            # Residuals plot
            residuals = y1 - y_pred
            axes[1].scatter(x1, residuals, color="purple")
            axes[1].axhline(0, color="black", linestyle="--")
            axes[1].set_title("Residuals vs X")
            axes[1].set_xlabel("x")
            axes[1].set_ylabel("Residuals")

            # Q-Q plot
            ordered_residuals = np.sort(residuals)
            theoretical_quants = norm.ppf((np.arange(1, n+1) - 0.5) / n)
            axes[2].scatter(ordered_residuals,theoretical_quants,  color="orange")

            #stats.probplot(residuals, dist="norm", plot=axes[2])
            axes[2].set_title("Normal Probability Plot")
            axes[2].set_xlabel("Residuals")
            axes[2].set_ylabel("Normal Probability Quantiles")
            
            
        
            plt.tight_layout()

            # Save to base64
            img = io.BytesIO()
            plt.savefig(img, format="png")
            img.seek(0)
            plot_url = base64.b64encode(img.getvalue()).decode()
            plt.close()
            values = pd.DataFrame({'x̄': [x_bar], 'ȳ': [y_bar], 'Sxx': [Sxx], 'Syy': [Syy], 'Sxy': [Sxy],
                                    'm': [m], 'b': [b], 'R_squared': [R_squared], 'sig': [sig], 
                                    'se': [se], 'SSE': [SSE]}).to_html(index=False, classes="fancy-table")

        

    return render_template("linear_descriptive.html", equation=equation, values = values, plot_url=plot_url, title="Linear Regression")


@linear_bp.route("/linearinferential")
def inferential():
    return render_template("linear_inferential.html", title="Linear Regression: Inferential Approach")



