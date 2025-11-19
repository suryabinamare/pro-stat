import base64

import io
from flask import Flask, request, Blueprint, render_template, current_app, jsonify
import os
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import f
from scipy.stats import norm
from scipy.stats import t
from scipy.stats import chi2
from werkzeug.utils import secure_filename
import matplotlib
matplotlib.use('Agg')  # Use non-GUI backend






distributions_bp = Blueprint("distributions", __name__, template_folder="../templates")


def create_normal_plot(mean, std):
    x = np.linspace(mean - 4*std, mean + 4*std, 400)
    y = (1/(std * np.sqrt(2*np.pi))) * np.exp(-((x-mean)**2) / (2*std**2))

    fig, ax = plt.subplots()
    ax.plot(x, y)
    ax.set_title(f"Normal Distribution (mean={mean}, std={std})")

    # Save plot to memory buffer
    buf = io.BytesIO()
    plt.savefig(buf, format="png")
    buf.seek(0)
    encoded = base64.b64encode(buf.read()).decode("utf-8")
    plt.close()
    return encoded



@distributions_bp.route("/zdistribution", methods=["GET", "POST"])
def zdistribution():
    z_value = None
    p_value = None
    plot_path = None
    error = None

    if request.method == 'POST':
        try:
            # Get form inputs
            z_input = request.form.get('z_given')
            p_input = request.form.get('p_given')

            if z_input:
                z_value = float(z_input)
                p_value = norm.cdf(z_value)

            elif p_input:
                p_value = float(p_input)
                if 0 < p_value < 1:
                    z_value = norm.ppf(p_value)
                else:
                    error = "P-value must be between 0 and 1 (exclusive)."
            else:
                error = "Please provide either a Z-score or a P-value."

            # === Generate Plot if we have a valid Z-score ===
            if z_value is not None and error is None:
                x = np.linspace(-3, 3, 500)
                y = norm.pdf(x, 0, 1)
                y1 = np.zeros(len(x))
                fig, ax = plt.subplots()

                ax.plot(x, y, label='Normal Distribution')
                ax.plot(z_value, 0, 'ro', label='z-score')
                ax.plot(x, y1, color='black')  # x-axis

                # Fill area under curve up to Z
                x_fill = np.linspace(-3, z_value, 500)
                y_fill = norm.pdf(x_fill, 0, 1)
                ax.fill_between(x_fill, y_fill, color='skyblue', alpha=0.6, label='P-value')

                ax.axvline(z_value, color='red', linestyle='--')
                ax.text(z_value - 0.1, 0.05, f'P-value = {p_value:.4f}', color='red', ha='right')
                #ax.set_title('Z-Score and P-Value on Normal Curve')
                ax.set_xlabel('Z-value')
                ax.set_ylabel('Probability Density')
                ax.set_xticks([z_value, 0])
                ax.legend()

                if not os.path.exists('static'):
                    os.makedirs('static')
                plot_path = 'static/plot_z.png'
                plt.tight_layout()
                plt.savefig(plot_path)
                plt.close()

        except Exception as e:
            error = f"Error: {str(e)}"

    return render_template('z-distribution.html',
                           z_value=z_value,
                           p_value=p_value,
                           plot_path=plot_path,
                           error=error)



@distributions_bp.route("/plot", methods = ['GET', 'POST'])
def plot():
    data = request.json
    mean = float(data["mean"])
    std = float(data["std"])

    image = create_normal_plot(mean, std)
    return jsonify({"image": image})






@distributions_bp.route("/tdistribution", methods=["GET", "POST"])

def tdistribution():
    t_value = None
    p_value = None
    df_input = None
    plot_path1 = None
    error = None

    if request.method == 'POST':
        try:
            # Get form inputs
            t_input = request.form.get('t_given')
            df_input = float(request.form.get('df'))
            p_input = request.form.get('p_given1')

            if t_input and df_input:
                if int(df_input):
                    t_value = float(t_input)
                    p_value = t.cdf(t_value, df_input)
                else:
                    error = "Degree of Freedom should be positive integer."

            elif p_input and df_input:
                p_value = float(p_input)
                if 0 < p_value < 1:
                    if int(df_input)>0:
                        t_value = t.ppf(p_value, df_input)
                    else:
                        error = "Degree of Freedom should be positive integer."
                else:
                    error = "P-value must be between 0 and 1 (exclusive)."
            else:
                error = "Please provide either a t-score and df or a P-value and df."

            # === Generate Plot if we have a valid t-score ===
            if t_value is not None and error is None:
                x = np.linspace(-3, 3, 500)
                y = t.pdf(x,df_input)
                y1 = np.zeros(len(x))
                fig, ax = plt.subplots(figsize = (6,4))

                ax.plot(x, y, label='t-Distribution')
                ax.plot(t_value, 0, 'ro', label='t-score')
                ax.plot(x, y1, color='black')  # x-axis


                # Fill area under curve up to Z
                x_fill = np.linspace(-3, t_value, 500)
                y_fill = t.pdf(x_fill, df_input)
                ax.fill_between(x_fill, y_fill, color='skyblue', alpha=0.6, label = "P-Value")

                ax.axvline(t_value, color='red', linestyle='--')
                ax.text(t_value - 0.1, 0.05, f'p-value = {p_value:.4f}', color='red', ha='right')
                #ax.set_title('Z-Score and P-Value on Normal Curve')
                ax.set_xlabel('t-value')
                ax.set_ylabel('Probability Density')
                ax.set_xticks([t_value])
                ax.legend()

                if not os.path.exists('static'):
                    os.makedirs('static')
                plot_path1 = 'static/plot_t.png'
                plt.tight_layout()
                plt.savefig(plot_path1)
                plt.close()

        except Exception as e:
            error = f"Error: {str(e)}"

    return render_template('t-distribution.html',
                           t_value=t_value,
                           df = df_input,
                           p_value=p_value,
                           plot_path1=plot_path1,
                           error=error)

    









# F-distribution

@distributions_bp.route("/fdistribution", methods=["GET", "POST"])
def fdistribution():
    f_value = None
    plot_fcurve = None 
    param = None 
    if request.method == 'POST':
        # check which form was submitted
        try:
            f_value = float(request.form.get('f_value', '').strip() or 0)
            df_num = int(request.form.get('df_num', '').strip() or 0)
            df_deno = int(request.form.get('df_deno', '').strip() or 0)
            alpha = float(request.form.get('alpha', '').strip() or 0)
            param = [df_num, df_deno, alpha]

            if df_num and df_deno and alpha:
                if df_deno < 1 and df_num < 1:
                    return render_template('F-distribution.html', p_value="Degrees of freedom must be positive integers.", f_value=None)
                elif alpha <= 0 or alpha >= 1:
                    return render_template('F-distribution.html', p_value="Alpha must be between 0 and 1 (exclusive).", f_value=None)
                else:
                    f_value = f.ppf(1 - alpha, df_num, df_deno)

                    #plotting density curve
                    mean = f.mean(df_num, df_deno)
                    std = f.std(df_num, df_deno)
                    x_max = mean + std*5       # denominator degrees of freedom

                    # Generate x values
                    x = np.linspace(0, x_max, 500)

                    # PDF values of the F-distribution
                    y = f.pdf(x, df_num, df_deno)
                    y1 = np.zeros(len(x))

                    # Plot
                    plt.figure(figsize=(6, 4))
                    plt.plot(x, y, label=f'F-dist (dfn={df_num}, dfd={df_deno})', color='blue')
                    plt.plot(x, y1, color='black')  # x-axis
                    plt.plot(f_value, 0, 'ro', label=f'F-value = {f_value:.4f}', markersize=8)
                    x_fill = np.linspace(f_value, x_max, 500)
                    y_fill = f.pdf(x_fill, df_num, df_deno)
                    #plt.fill_between(x, y, alpha=0.2, color='skyblue')
                
                    plt.scatter(f_value, 0, color='red', s=100, zorder=5)  # Dot at (x_value, 0)
                    plt.fill_between(x_fill, y_fill, alpha = 0.3, color = 'blue', label=f'p-value, (α={alpha})' )
                    plt.axvline(f_value, linestyle = '--', color = 'red')
                    plt.title("F-Distribution Density Curve")
                    plt.xlabel("F-value")
                    plt.ylabel("Density")
                    plt.grid(True, linestyle='--', alpha=0.5)
                    plt.legend()
                    if not os.path.exists('static'):
                        os.makedirs('static')
                    plot_fcurve = 'static/fcurve.png'
                    plt.tight_layout()
                    plt.savefig(plot_fcurve)
                    plt.close()
            elif f_value and df_num and df_deno:
                if df_deno < 1 and df_num < 1:
                    return render_template('F-distribution.html', p_value="Degrees of freedom must be positive integers.", f_value=None)
                elif f_value <= 0:
                    return render_template('F-distribution.html', p_value="F-value must be positive.", f_value=None)
                else:
                    alpha = 1 - f.cdf(f_value, df_num, df_deno)

                    param = [df_num, df_deno, alpha]
                    #plotting density curve
                    mean = f.mean(df_num, df_deno)
                    std = f.std(df_num, df_deno)
                    x_max = mean + std*5       # denominator degrees of freedom

                    # Generate x values
                    x = np.linspace(0, x_max, 500)

                    # PDF values of the F-distribution
                    y = f.pdf(x, df_num, df_deno)
                    y1 = np.zeros(len(x))

                    # Plot
                    plt.figure(figsize=(6, 4))
                    plt.plot(x, y, label=f'F-dist (dfn={df_num}, dfd={df_deno})', color='blue')
                    plt.plot(x, y1, color='black')  # x-axis
                    plt.plot(f_value, 0, 'ro', label=f'F-value = {f_value:.4f}', markersize=8)
                    x_fill = np.linspace(f_value, x_max, 500)
                    y_fill = f.pdf(x_fill, df_num, df_deno)
                    #plt.fill_between(x, y, alpha=0.2, color='skyblue')
                
                    plt.scatter(f_value, 0, color='red', s=100, zorder=5)  # Dot at (x_value, 0)
                    plt.fill_between(x_fill, y_fill, alpha = 0.6, color = 'blue', label=f'p-value, (α={alpha})' )
                    plt.axvline(f_value, linestyle = '--', color = 'red')
                    plt.title("F-Distribution Density Curve")
                    plt.xlabel("F-value")
                    plt.ylabel("Density")
                    plt.grid(True, linestyle='--', alpha=0.5)
                    plt.legend()
                    if not os.path.exists('static'):
                        os.makedirs('static')
                    plot_fcurve = 'static/fcurve.png'
                    plt.tight_layout()
                    plt.savefig(plot_fcurve)
                    plt.close()
            else:
                return render_template('F-distribution.html', p_value="Please provide all required inputs.", f_value=None)
        except ValueError:
            return render_template('F-distribution.html', p_value="Invalid input", f_value=None)

    return render_template(
        'F-distribution.html',
        f_value=f_value,
        plot_fcurve=plot_fcurve,
        param = param,
        zip=zip,
        enumerate=enumerate)











@distributions_bp.route("/chisquaredistribution", methods=["GET", "POST"])
def chisquaredistribution():
    chisquare = None
    p_value = None
    df_input = None
    plot_path_chisquare = None
    error = None

    if request.method == 'POST':
        try:
            # Get form inputs
            chisquare = request.form.get('chi-square')
            df_input = float(request.form.get('df'))
            p_input = request.form.get('p-value')

            if chisquare and df_input:
                if int(df_input):
                    chisquare = float(chisquare)
                    p_value = chi2.sf(chisquare, df_input)
                    x = np.linspace(0, 5*(df_input), 500)
                    y = chi2.pdf(x, df_input)
                    y1 = np.zeros(len(x))
                    fig, ax = plt.subplots(figsize = (6,4))

                    ax.plot(x, y, label='$\chi^2$ Distribution')
                    ax.plot(chisquare, 0, 'ro', label='$\chi^2$ - Score')
                    ax.plot(x, y1, color='black')  # x-axis


                    # Fill area under curve up to Z
                    x_fill = np.linspace(chisquare, 5*(df_input), 500)
                    y_fill = chi2.pdf(x_fill, df_input)
                    ax.fill_between(x_fill, y_fill, color='skyblue', alpha=0.6, label = "P-Value")

                    ax.axvline(chisquare, color='red', linestyle='--')
                    ax.text(chisquare + 0.5, 0.05, f'p-value = {p_value:.4f}', color='red', ha='left')
                    #ax.set_title('Z-Score and P-Value on Normal Curve')
                    ax.set_xlabel('$\chi^2$ Value')
                    ax.set_ylabel('Probability Density')
                    ax.set_xticks([0,chisquare])
                    ax.legend()

                    if not os.path.exists('static'):
                        os.makedirs('static')
                    plot_path_chisquare = 'static/plot_chisquare.png'
                    plt.tight_layout()
                    plt.savefig(plot_path_chisquare)
                    plt.close()
                else:
                    error = "Degree of Freedom should be positive integer."

            elif p_input and df_input:
                p_value = float(p_input)
                if 0 < p_value < 1:
                    if int(df_input)>0:
                        chisquare = chi2.isf(p_value, df_input)
                        x = np.linspace(0, 5*(df_input), 500)
                        y = chi2.pdf(x, df_input)
                        y1 = np.zeros(len(x))
                        fig, ax = plt.subplots(figsize = (6,4))

                        ax.plot(x, y, label='$\chi^2$ Distribution')
                        ax.plot(chisquare, 0, 'ro', label='$\chi^2$ - Score')
                        ax.plot(x, y1, color='black')  # x-axis


                        # Fill area under curve up to Z
                        x_fill = np.linspace(chisquare, 5*(df_input), 500)
                        y_fill = chi2.pdf(x_fill, df_input)
                        ax.fill_between(x_fill, y_fill, color='skyblue', alpha=0.6, label = "P-Value")

                        ax.axvline(chisquare, color='red', linestyle='--')
                        ax.text(chisquare + 0.5, 0.05, f'p-value = {p_value:.4f}', color='red', ha='left')
                        #ax.set_title('Z-Score and P-Value on Normal Curve')
                        ax.set_xlabel('$\chi^2$ Value')
                        ax.set_ylabel('Probability Density')
                        ax.set_xticks([0, chisquare])
                        ax.legend()

                        if not os.path.exists('static'):
                            os.makedirs('static')
                        plot_path_chisquare = 'static/plot_chisquare.png'
                        plt.tight_layout()
                        plt.savefig(plot_path_chisquare)
                        plt.close()
                    else:
                        error = "Degree of Freedom should be positive integer."
                else:
                    error = "P-value must be between 0 and 1 (exclusive)."
            else:
                error = "Please provide either a chi-square score and df or a P-value and df."

        except Exception as e:
            error = f"Error: {str(e)}"

    return render_template('chisquare_distribution.html',
                           chisquare=chisquare,
                           df=df_input,
                           pvalue=p_value,
                           plot_path_chisquare=plot_path_chisquare,
                           error=error)

