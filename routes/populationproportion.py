from flask import Flask, request, Blueprint, render_template, current_app
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



populationproportion_bp = Blueprint("populationproportion", __name__)




# Confidence Interval for a population proportion. 
@populationproportion_bp.route("/confidence_interval", methods=["GET", "POST"])
def confidence_interval():
    ci_lower = None
    ci_upper = None
    margin_of_error = None
    z_alpha = None
    p_hat = None
    n = None
    alpha = None
    standard_error = None
    if request.method == 'POST':
        try:
            p_hat = float(request.form.get('sample_proportion', 0.5))
            n = int(request.form.get('sample_size', 30))
            alpha = float(request.form.get('alpha', 0.05))
            z_alpha = norm.ppf(1 - alpha / 2)
            margin_of_error = z_alpha * np.sqrt((p_hat * (1 - p_hat)) / n)
            ci_lower = p_hat - margin_of_error
            ci_upper = p_hat + margin_of_error
        except ValueError:
            return "Invalid input. Please enter valid numbers."
        
    return render_template("popproportion_CI.html", 
                           ci_lower=ci_lower, ci_upper=ci_upper,
                           margin_of_error=margin_of_error, z_alpha=z_alpha,
                           sample_proportion=p_hat,
                             sample_size =n, alpha=alpha,
                            standard_error = np.sqrt((p_hat * (1 - p_hat)) / n) if n else None)





# Confidence Intervals for two Population Proportions
@populationproportion_bp.route("/confidence_interval_two", methods=["GET", "POST"])
def confidence_interval_two():
    ci_lower = None
    ci_upper = None
    margin_of_error = None
    z_alpha = None
    p1_hat = None
    p2_hat = None
    n1 = None
    n2 = None
    alpha = None
    standard_error = None
    if request.method == 'POST':
        try:
            p1_hat = float(request.form.get('sample_proportion1', 0.5))
            p2_hat = float(request.form.get('sample_proportion2', 0.5))
            n1 = int(request.form.get('sample_size1', 30))
            n2 = int(request.form.get('sample_size2', 30))
            alpha = float(request.form.get('alpha', 0.05))
            z_alpha = norm.ppf(1 - alpha / 2)
            standard_error = np.sqrt((p1_hat * (1 - p1_hat)) / n1 + (p2_hat * (1 - p2_hat)) / n2)
            margin_of_error = z_alpha * standard_error
            ci_lower = (p1_hat - p2_hat) - margin_of_error
            ci_upper = (p1_hat - p2_hat) + margin_of_error
        except ValueError:
            return "Invalid input. Please enter valid numbers."

    return render_template("popproportion_CI2.html",
                           ci_lower=ci_lower, ci_upper=ci_upper,
                           margin_of_error=margin_of_error, z_alpha=z_alpha,
                           sample_proportion1=p1_hat, sample_proportion2=p2_hat,
                           sample_size1=n1, sample_size2=n2, alpha=alpha,
                           standard_error=standard_error)

    










@populationproportion_bp.route("/simulate", methods=["GET", "POST"])
def sample_proportion_simulation():
    if request.method == 'POST':
        try:
            p = float(request.form.get('pop_proportion', 0.5))
            n = int(request.form.get('sample_size', 30))
            trials = int(request.form.get('num_samples', 100))
            sample_proportions = []
            for _ in range(trials):
                sample = np.random.binomial(1, p, n)
                prop = np.mean(sample)
                sample_proportions.append(prop)
        except ValueError:
            return "Invalid input. Please enter valid numbers."
        # Plot histogram
        plt.hist(sample_proportions, bins=15, edgecolor='black', alpha=0.7)
        plt.axvline(p, color='red', linestyle='dashed', linewidth=2, label=f"True p = {p}")
        plt.xlabel("Sample Proportion")
        plt.ylabel("Frequency")
        plt.title(f"Sampling Distribution of Sample Proportion (n={n}, trials={trials})")
        plt.legend()
        if not os.path.exists('static'):
            os.makedirs('static')
        plt.savefig('static/proportion_simulation.png')
        plt.close()
        return render_template("popproportion1.html")




@populationproportion_bp.route("/proportion1", methods=["GET", "POST"])
def proportion1():
    z = None
    p_value = None
    test_type = None
    z_alpha = None
    p_0 = None
    p_hat = None
    alpha = None
    n = None
    if request.method == 'POST':
        try:
            p_0 = float(request.form.get('pop_proportion', 0.5))
            p_hat = float(request.form.get('sample_proportion', 0.5))
            n = int(request.form.get('sample_size', 30))
            alpha= float(request.form.get('alpha', 0.05))
            test_type = request.form.get('Type')
            if test_type == "two-tailed":
                # Perform two-tailed test
                z = (p_hat - p_0) / np.sqrt((p_0 * (1 - p_0)) / n)
                z_alpha = norm.ppf(1 - alpha / 2)
                p_value = 2 * (1 - norm.cdf(abs(z)))
                # Create both subplots side by side
                fig, ax2 = plt.subplots(1, 1, figsize=(6, 4))


                # ===== Second Plot (Standard Normal Distribution) =====
                x1 = np.linspace(-3, 3, 500)
                y1 = norm.pdf(x1, loc=0, scale=1)
                y2 = np.zeros(500)

                ax2.plot(x1, y1)
                ax2.plot(x1, y2)
                ax2.plot(-z_alpha, 0, 'ro', label = f'crit_val = {-z_alpha:.2f}')
                ax2.plot(z_alpha, 0, 'ro', label = f'crit_val = {z_alpha:.2f}')
                ax2.plot(z, 0, 'go', label = f'test statistic:z = {z:.2f}')

                # Shade rejection regions
                x_fill_left1 = np.linspace(-z_alpha, -3, 500)
                y_fill_left1 = norm.pdf(x_fill_left1, loc=0, scale=1)

                x_fill_right1 = np.linspace(z_alpha, 3, 500)
                y_fill_right1 = norm.pdf(x_fill_right1, loc=0, scale=1)

                ax2.fill_between(x_fill_left1, y_fill_left1, color='skyblue', alpha=0.6)
                ax2.fill_between(x_fill_right1, y_fill_right1, color='skyblue', alpha=0.6)

                ax2.axvline(-z_alpha, color='red', linestyle='--')
                ax2.axvline(z_alpha, color='red', linestyle='--')

                ax2.text(0, 0.1, 'Do Not Reject $H_0$', fontsize=10, color='red', ha='center')
                ax2.text(-z_alpha - 0.1, 0.1, 'Reject $H_0$', fontsize=10, color='red', ha='right')
                ax2.text(z_alpha + 0.1, 0.1, 'Reject $H_0$', fontsize=10, color='red', ha='left')

                ax2.set_title('Standard Normal Distribution')
                ax2.set_xticks([-z_alpha, 0, z_alpha])
                ax2.set_xlabel('$z$ Values')
                ax2.legend(loc = 'upper right')

                # Save the combined figure
                plt.tight_layout()
                if not os.path.exists('static'):
                    os.makedirs('static')
                plt.savefig('static/prop_twosided.png')
                plt.close()
            
                

            elif test_type == "left-tailed":
                # Perform left-tailed test
                z = (p_hat - p_0) / np.sqrt((p_0 * (1 - p_0)) / n)
                z_alpha = norm.ppf(alpha)
                p_value = norm.cdf(z)
                # Create both subplots side by side
                fig, ax2 = plt.subplots(1, 1, figsize=(6, 4))


                # ===== Second Plot (Standard Normal Distribution) =====
                x1 = np.linspace(-3, 3, 500)
                y1 = norm.pdf(x1, loc=0, scale=1)
                y2 = np.zeros(500)

                ax2.plot(x1, y1)
                ax2.plot(x1, y2)
                ax2.plot(z_alpha, 0, 'ro', label = f'crit_val = {z_alpha:.2f}')
                ax2.plot(z, 0, 'go', label = f'test statistic:z = {z:.2f}')
            

                # Shade rejection regions
                x_fill_left1 = np.linspace(-3, z_alpha, 500)
                y_fill_left1 = norm.pdf(x_fill_left1, loc=0, scale=1)


                ax2.fill_between(x_fill_left1, y_fill_left1, color='skyblue', alpha=0.6)
            

                ax2.axvline(z_alpha, color='red', linestyle='--')

    
                ax2.text(z_alpha - 0.1, 0.1, 'Reject $H_0$', fontsize=10, color='red', ha='right')
                ax2.text(z_alpha + 0.1, 0.1, 'Do not Reject $H_0$', fontsize=10, color='red', ha='left')

                ax2.set_title('Standard Normal Distribution')
                ax2.set_xticks([z_alpha, 0, -z_alpha])
                ax2.set_xlabel('$z$ Values')
                ax2.legend()

                # Save the combined figure
                plt.tight_layout()
                if not os.path.exists('static'):
                    os.makedirs('static')
                plt.savefig('static/prop_twosided.png')
                plt.close()
                
            elif test_type == "right-tailed":
                # Perform right-tailed test
                z = (p_hat - p_0) / np.sqrt((p_0 * (1 - p_0)) / n)
                z_alpha = norm.ppf(1 - alpha)
                p_value = 1 - norm.cdf(z)
                # Create both subplots side by side
                fig, ax2 = plt.subplots(1, 1, figsize=(6, 4))

                # ===== Second Plot (Standard Normal Distribution) =====
                x1 = np.linspace(-3, 3, 500)
                y1 = norm.pdf(x1, loc=0, scale=1)
                y2 = np.zeros(500)

                ax2.plot(x1, y1)
                ax2.plot(x1, y2)
                ax2.plot(z_alpha, 0, 'ro', label = f'crit_val = {z_alpha:.2f}')
                ax2.plot(z, 0, 'go', label = f'test statistic:z = {z:.2f}')

                # Shade rejection regions
                x_fill_left1 = np.linspace(z_alpha,3, 500)
                y_fill_left1 = norm.pdf(x_fill_left1, loc=0, scale=1)

                ax2.fill_between(x_fill_left1, y_fill_left1, color='skyblue', alpha=0.6)

                ax2.axvline(z_alpha, color='red', linestyle='--')

                ax2.text(z_alpha - 0.1, 0.1, 'Do not Reject $H_0$', fontsize=10, color='red', ha='right')
                ax2.text(z_alpha + 0.1, 0.1, 'Reject $H_0$', fontsize=10, color='red', ha='left')

                ax2.set_title('Standard Normal Distribution')
                ax2.set_xticks([-z_alpha, 0, z_alpha])
                ax2.set_xlabel('$z$ Values')
                ax2.legend()

                # Save the combined figure
                plt.tight_layout()
                if not os.path.exists('static'):
                    os.makedirs('static')
                plt.savefig('static/prop_twosided.png')
                plt.close()
        except ValueError:
            return "Invalid input. Please enter valid numbers."
    return render_template("popproportion1.html", z=z, p_value=p_value, z_alpha=z_alpha, 
                           pop_proportion = p_0, sample_proportion = p_hat, alpha = alpha, sample_size = n, test_type=test_type)





@populationproportion_bp.route("/proportion2", methods=["GET", "POST"])
def proportion2():
    z = None
    p_value = None
    test_type = None
    z_alpha = None
    alpha = None
    p_hat1 = None
    p_hat2 = None   
    n1 = None
    n2 = None
    x1 = None
    x2 = None
    p = None
    if request.method == 'POST':
        try:
            n1 = int(request.form.get('sample1', 30))
            n2 = int(request.form.get('sample2', 30))
            x1 = float(request.form.get('success1', 0.5))
            x2 = float(request.form.get('success2', 0.5))
            p_hat1 = float(request.form.get('sample_proportion1', 0.5))
            p_hat2 = float(request.form.get('sample_proportion2', 0.5))            
            alpha= float(request.form.get('alpha', 0.05))
            test_type = request.form.get('Type')
            p = (x1 + x2) / (n1 + n2)
            z = (p_hat1 - p_hat2) / np.sqrt(p * (1 - p) * (1/n1 + 1/n2))

            if test_type == "two-tailed":
                # Perform two-tailed test
                z_alpha = norm.ppf(1 - alpha / 2)
                p_value = 2 * (1 - norm.cdf(abs(z)))

                # Create plots:
                fig, ax2 = plt.subplots(1, 1, figsize=(6, 4))
                x = np.linspace(-3, 3, 500)
                y = norm.pdf(x, loc=0, scale=1)
                y1 = np.zeros(500)

                ax2.plot(x, y)
                ax2.plot(-z_alpha, 0, 'ro', label = 'Critical Value')
                ax2.plot(z_alpha, 0, 'ro', label = 'Critical Value')
                ax2.plot(z, 0, 'go', label = 'Test Statistic')
                ax2.plot(x, y1)

                # Shade rejection regions
                x_fill_left1 = np.linspace(-z_alpha, -3, 500)
                y_fill_left1 = norm.pdf(x_fill_left1, loc=0, scale=1)

                x_fill_right1 = np.linspace(z_alpha, 3, 500)
                y_fill_right1 = norm.pdf(x_fill_right1, loc=0, scale=1)

                ax2.fill_between(x_fill_left1, y_fill_left1, color='skyblue', alpha=0.6)
                ax2.fill_between(x_fill_right1, y_fill_right1, color='skyblue', alpha=0.6)

                ax2.axvline(-z_alpha, color='red', linestyle='--')
                ax2.axvline(z_alpha, color='red', linestyle='--')

                ax2.text(0, 0.1, 'Do Not Reject $H_0$', fontsize=10, color='red', ha='center')
                ax2.text(-z_alpha - 0.1, 0.1, 'Reject $H_0$', fontsize=10, color='red', ha='right')
                ax2.text(z_alpha + 0.1, 0.1, 'Reject $H_0$', fontsize=10, color='red', ha='left')

                ax2.set_title('Standard Normal Distribution')
                ax2.set_xticks([-z_alpha, 0, z_alpha])
                ax2.set_xlabel('$z$ Values')
                ax2.legend()

                # Save the combined figure
                plt.tight_layout()
                if not os.path.exists('static'):
                    os.makedirs('static')
                plt.savefig('static/twoproportions.png')
                plt.close()
            
                

            elif test_type == "left-tailed":
                # Perform left-tailed test
                z_alpha = norm.ppf(alpha)
                p_value = norm.cdf(z)
                # Create both subplots side by side
                fig, ax2 = plt.subplots(1, 1, figsize=(6, 4))


                # ===== Second Plot (Standard Normal Distribution) =====
                x = np.linspace(-3, 3, 500)
                y = norm.pdf(x, loc=0, scale=1)
                y2 = np.zeros(500)

                ax2.plot(x, y)
                ax2.plot(z_alpha, 0, 'ro', label = 'Critical Value')
                ax2.plot(z, 0, 'go', label = 'Test Statistic')
                ax2.plot(x, y2)

                # Shade rejection regions
                x_fill_left1 = np.linspace(-3, z_alpha, 500)
                y_fill_left1 = norm.pdf(x_fill_left1, loc=0, scale=1)


                ax2.fill_between(x_fill_left1, y_fill_left1, color='skyblue', alpha=0.6)
            

                ax2.axvline(z_alpha, color='red', linestyle='--')

    
                ax2.text(z_alpha - 0.1, 0.1, 'Reject $H_0$', fontsize=10, color='red', ha='right')
                ax2.text(z_alpha + 0.1, 0.1, 'Do not Reject $H_0$', fontsize=10, color='red', ha='left')

                ax2.set_title('Standard Normal Distribution')
                ax2.set_xticks([z_alpha, 0, -z_alpha])
                ax2.set_xlabel('$z$ Values')
                ax2.legend()

                # Save the combined figure
                plt.tight_layout()
                if not os.path.exists('static'):
                    os.makedirs('static')
                plt.savefig('static/twoproportions.png')
                plt.close()
                
            elif test_type == "right-tailed":
                # Perform right-tailed test
                z_alpha = norm.ppf(1 - alpha)
                p_value = 1 - norm.cdf(z)
                # Create both subplots side by side
                fig, ax2 = plt.subplots(1, 1, figsize=(6, 4))

                # ===== Second Plot (Standard Normal Distribution) =====
                x = np.linspace(-3, 3, 500)
                y = norm.pdf(x, loc=0, scale=1)
                y2 = np.zeros(500)

                ax2.plot(x, y)
                ax2.plot(z_alpha, 0, 'ro', label = 'Critical Value')
                ax2.plot(z, 0, 'go', label = 'Test Statistic')
                ax2.plot(x, y2)

                # Shade rejection regions
                x_fill_left1 = np.linspace(z_alpha,3, 500)
                y_fill_left1 = norm.pdf(x_fill_left1, loc=0, scale=1)

                ax2.fill_between(x_fill_left1, y_fill_left1, color='skyblue', alpha=0.6)

                ax2.axvline(z_alpha, color='red', linestyle='--')

                ax2.text(z_alpha - 0.1, 0.1, 'Do not Reject $H_0$', fontsize=10, color='red', ha='right')
                ax2.text(z_alpha + 0.1, 0.1, 'Reject $H_0$', fontsize=10, color='red', ha='left')

                ax2.set_title('Standard Normal Distribution')
                ax2.set_xticks([-z_alpha, 0, z_alpha])
                ax2.set_xlabel('$z$ Values')
                ax2.legend()

                # Save the combined figure
                plt.tight_layout()
                if not os.path.exists('static'):
                    os.makedirs('static')
                plt.savefig('static/twoproportions.png')
                plt.close()
        except ValueError:
            return "Invalid input. Please enter valid numbers."
    return render_template("popproportion2.html", z=z, p_value=p_value, z_alpha=z_alpha, 
                           sample1 = n1, sample2 = n2, success1 = x1, success2 = x2,
                            sample_proportion1 = p_hat1, sample_proportion2 = p_hat2,
                            alpha = alpha, p = p, test_type=test_type)





