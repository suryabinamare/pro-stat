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
import matplotlib
import math
matplotlib.use('Agg')  # Use non-GUI backend


populationmean_bp = Blueprint("populationmean", __name__, template_folder="../templates")










@populationmean_bp.route("/sigmaknown", methods=["GET", "POST"])
def sigmaknown():
    values = None
    z = None
    z_test = None
    p_value = None
    test_type = None
    CI = None
    if request.method == 'POST':
        try:
            sample_mean = float(request.form['sample_mean'])
            pop_mean = float(request.form['population_mean'])
            sample_size = int(request.form['sample_size'])
            alpha = float(request.form['alpha'])
            sigma = float(request.form['sigma'])
            test_type = request.form['Type']
            z_test = (sample_mean-pop_mean)/(sigma/np.sqrt(sample_size))
            if test_type == "two-tailed":
                z = norm.ppf(1 - alpha / 2)
                p_value = 2 * (1 - norm.cdf(abs(z_test)))
                CI = (sample_mean - z * sigma / np.sqrt(sample_size), sample_mean + z * sigma / np.sqrt(sample_size))

                values = (sample_mean, pop_mean, sample_size, sigma, alpha, z_test, z, p_value)

                # Create both subplots side by side
                fig, ax1 = plt.subplots(1, 1, figsize=(6, 4))

                # ===== First Plot (Population Mean Distribution) =====
                x = np.linspace(-4, 4, 500)
                y = norm.pdf(x, 0, 1)
                y1 = np.zeros(500)

                ax1.plot(x, y)
                ax1.plot(-z, 0, 'ro', label = 'Critical Value')
                ax1.plot(z, 0, 'ro', label = 'Critical Value')
                ax1.plot(z_test, 0, 'bo', label = 'Z Test Statistic')
                ax1.plot(x, y1)


                # Fill rejection regions
                x_fill_left = np.linspace(-4, - z, 500)
                y_fill_left = norm.pdf(x_fill_left, 0, 1)

                x_fill_right = np.linspace(z, 4, 500)
                y_fill_right = norm.pdf(x_fill_right, 0, 1)

                ax1.fill_between(x_fill_left, y_fill_left, color='skyblue', alpha=0.6)
                ax1.fill_between(x_fill_right, y_fill_right, color='skyblue', alpha=0.6)

                ax1.axvline(-z, color='red', linestyle='--')
                ax1.axvline(z, color='red', linestyle='--')

                ax1.text(0, 0.1, 'Do Not Reject $H_0$', fontsize=10, color='red', ha='center')
                ax1.text(- z - 0.1, 0.1, 'Reject $H_0$', fontsize=10, color='red', ha='right')
                ax1.text(z + 0.1, 0.1, 'Reject $H_0$', fontsize=10, color='red', ha='left')

                ax1.set_title('Rejection/Non-rejection Region')
                ax1.set_xticks([-z, 0, z])
                ax1.set_xlabel('$z$ Values')
                ax1.legend()

                # Save the combined figure
                plt.tight_layout()
                if not os.path.exists('static'):
                    os.makedirs('static')
                plt.savefig('static/plot.png')
                plt.close()

            elif test_type == "left-tailed":
                z = norm.ppf(alpha)
                p_value = norm.cdf(z_test)

                values = (sample_mean, pop_mean, sample_size, sigma, alpha, z_test, z, p_value)
                CI = (None, sample_mean + z * sigma / np.sqrt(sample_size)) #lower confidence bound

                # Create both subplots side by side
                fig, ax1 = plt.subplots(1, 1, figsize=(6, 4))

                # ===== First Plot (Population Mean Distribution) =====
                x = np.linspace(-4, 4, 500)
                y = norm.pdf(x, 0, 1)
                y1 = np.zeros(500)

                ax1.plot(x, y)
                ax1.plot(z, 0, 'ro', label = 'Critical Value')
        
                ax1.plot(z_test, 0, 'bo', label = 'Z Test Statistic')
                ax1.plot(x, y1)

              # Fill rejection regions
                x_fill_left = np.linspace(-4, z, 500)
                y_fill_left = norm.pdf(x_fill_left, 0, 1)

                ax1.fill_between(x_fill_left, y_fill_left, color='skyblue', alpha=0.6)

                ax1.axvline(z, color='red', linestyle='--')
              
                ax1.text(z - 0.15, 0.12, 'Reject $H_0$', fontsize=10, color='red', ha='right')
                ax1.text(z + 0.15, 0.12, 'Do Not Reject $H_0$', fontsize=10, color='red', ha='left')

                ax1.set_title('Rejection/Non-rejection Region')
                ax1.set_xticks([z, 0])
                ax1.set_xlabel('$z$ Values')
                ax1.legend()

                # Save the combined figure
                plt.tight_layout()
                if not os.path.exists('static'):
                    os.makedirs('static')
                plt.savefig('static/plot.png')
                plt.close()
            elif test_type == "right-tailed":
                z = norm.ppf(1-alpha)
                p_value = 1 - norm.cdf(z_test)
                values = (sample_mean, pop_mean, sample_size, sigma, alpha, z_test, z, p_value)
                CI = (sample_mean - z * sigma / np.sqrt(sample_size), None) #upper confidence bound

                # Create both subplots side by side
                fig, ax1 = plt.subplots(1, 1, figsize=(6, 4))

                # ===== First Plot (Population Mean Distribution) =====
                x = np.linspace(-4, 4, 500)
                y = norm.pdf(x, 0, 1)
                y1 = np.zeros(500)

                ax1.plot(x, y)
                ax1.plot(z, 0, 'ro', label = 'Critical Value')
        
                ax1.plot(z_test, 0, 'bo', label = 'Z Test Statistic')
                ax1.plot(x, y1)

              # Fill rejection regions
                x_fill_left = np.linspace(z,4, 500)
                y_fill_left = norm.pdf(x_fill_left, 0, 1)

                ax1.fill_between(x_fill_left, y_fill_left, color='skyblue', alpha=0.6)

                ax1.axvline(z, color='red', linestyle='--')
              
                ax1.text(z - 0.15, 0.12, ' Do Not Reject $H_0$', fontsize=10, color='red', ha='right')
                ax1.text(z + 0.15, 0.12, 'Reject $H_0$', fontsize=10, color='red', ha='left')

                ax1.set_title('Rejection/Non-rejection Region')
                ax1.set_xticks([0, z])
                ax1.set_xlabel('$z$ Values')
                ax1.legend()

                # Save the combined figure
                plt.tight_layout()
                if not os.path.exists('static'):
                    os.makedirs('static')
                plt.savefig('static/plot.png')
                plt.close()

        except Exception as e:
            ci_result = f"Error: {e}"

    return render_template('popmean_sigmaknown.html', values = values, CI = CI,test_type = test_type)








@populationmean_bp.route("/sigmaunknown", methods=["GET", "POST"])
def sigmaunknown():
    values = None
    test_type = None
    CI = None
    df = None
    if request.method == 'POST':
        try:
            sample_mean = float(request.form['sample_mean'])
            pop_mean = float(request.form['population_mean'])
            sample_size = int(request.form['sample_size'])
            alpha = float(request.form['alpha'])
            sigma = float(request.form['s'])
            df = int(request.form['df'])
            test_type = request.form['Type']
            t_test = (sample_mean-pop_mean)/(sigma/np.sqrt(sample_size))
            if test_type == "two-tailed":
                t_crit = t.ppf(1 - alpha / 2, df)
                p_value = 2 * (1 - t.cdf(abs(t_test), df))
                CI = (sample_mean - t_crit * sigma / np.sqrt(sample_size), sample_mean + t_crit * sigma / np.sqrt(sample_size))

                values = (sample_mean, pop_mean, sample_size, sigma, alpha, t_test, t_crit, p_value)

                
                fig, ax1 = plt.subplots(1, 1, figsize=(6, 4))

                # ===== First Plot (Population Mean Distribution) =====
                x = np.linspace(-4, 4, 500)
                y = t.pdf(x, df)
                y1 = np.zeros(500)

                ax1.plot(x, y)
                ax1.plot(-t_crit, 0, 'ro', label = 'Critical Value')
                ax1.plot(t_crit, 0, 'ro', label = 'Critical Value')
                ax1.plot(t_test, 0, 'bo', label = 'T Test Statistic')
                ax1.plot(x, y1)


                # Fill rejection regions
                x_fill_left = np.linspace(-4, - t_crit, 500)
                y_fill_left = t.pdf(x_fill_left, df)

                x_fill_right = np.linspace(t_crit, 4, 500)
                y_fill_right = t.pdf(x_fill_right, df)

                ax1.fill_between(x_fill_left, y_fill_left, color='skyblue', alpha=0.6)
                ax1.fill_between(x_fill_right, y_fill_right, color='skyblue', alpha=0.6)

                ax1.axvline(-t_crit, color='red', linestyle='--')
                ax1.axvline(t_crit, color='red', linestyle='--')

                ax1.text(0, 0.1, 'Do Not Reject $H_0$', fontsize=10, color='red', ha='center')
                ax1.text(- t_crit - 0.1, 0.1, 'Reject $H_0$', fontsize=10, color='red', ha='right')
                ax1.text(t_crit + 0.1, 0.1, 'Reject $H_0$', fontsize=10, color='red', ha='left')

                ax1.set_title('Rejection/Non-rejection Region')
                ax1.set_xticks([-t_crit, 0, t_crit])
                ax1.set_xlabel('$t$ Values')
                ax1.legend()

                # Save the combined figure
                plt.tight_layout()
                if not os.path.exists('static'):
                    os.makedirs('static')
                plt.savefig('static/plot_sigmaunknown.png')
                plt.close()

            elif test_type == "left-tailed":
                t_crit = t.ppf(alpha, df)
                p_value = t.cdf(t_test, df)

                values = (sample_mean, pop_mean, sample_size, sigma, alpha, t_test, t_crit, p_value)
                CI = (None, sample_mean + t_crit * sigma / np.sqrt(sample_size)) #upper confidence bound

                # Create both subplots side by side
                fig, ax1 = plt.subplots(1, 1, figsize=(6, 4))

                # ===== First Plot (Population Mean Distribution) =====
                x = np.linspace(-4, 4, 500)
                y = t.pdf(x, df)
                y1 = np.zeros(500)

                ax1.plot(x, y)
                ax1.plot(t_crit, 0, 'ro', label = 'Critical Value')
        
                ax1.plot(t_test, 0, 'bo', label = 't Test Statistic')
                ax1.plot(x, y1)

              # Fill rejection regions
                x_fill_left = np.linspace(-4, t_crit, 500)
                y_fill_left = t.pdf(x_fill_left, df)

                ax1.fill_between(x_fill_left, y_fill_left, color='skyblue', alpha=0.6)

                ax1.axvline(t_crit, color='red', linestyle='--')

                ax1.text(t_crit - 0.15, 0.12, 'Reject $H_0$', fontsize=10, color='red', ha='right')
                ax1.text(t_crit + 0.15, 0.12, 'Do Not Reject $H_0$', fontsize=10, color='red', ha='left')

                ax1.set_title('Rejection/Non-rejection Region')
                ax1.set_xticks([t_crit, 0])
                ax1.set_xlabel('$t$ Values')
                ax1.legend()

                # Save the combined figure
                plt.tight_layout()
                if not os.path.exists('static'):
                    os.makedirs('static')
                plt.savefig('static/plot_sigmaunknown.png')
                plt.close()
            elif test_type == "right-tailed":
                t_crit = t.ppf(1-alpha, df)
                p_value = 1 - t.cdf(t_test, df)
                values = (sample_mean, pop_mean, sample_size, sigma, alpha, t_test, t_crit, p_value)
                CI = (sample_mean - t_crit * sigma / np.sqrt(sample_size), None) #upper confidence bound

                # Create both subplots side by side
                fig, ax1 = plt.subplots(1, 1, figsize=(6, 4))

                # ===== First Plot (Population Mean Distribution) =====
                x = np.linspace(-4, 4, 500)
                y = t.pdf(x, df)
                y1 = np.zeros(500)

                ax1.plot(x, y)
                ax1.plot(t_crit, 0, 'ro', label = 'Critical Value')

                ax1.plot(t_test, 0, 'bo', label = 't Test Statistic')
                ax1.plot(x, y1)

              # Fill rejection regions
                x_fill_left = np.linspace(t_crit, 4, 500)
                y_fill_left = t.pdf(x_fill_left, df)

                ax1.fill_between(x_fill_left, y_fill_left, color='skyblue', alpha=0.6)

                ax1.axvline(t_crit, color='red', linestyle='--')

                ax1.text(t_crit - 0.15, 0.12, ' Do Not Reject $H_0$', fontsize=10, color='red', ha='right')
                ax1.text(t_crit + 0.15, 0.12, 'Reject $H_0$', fontsize=10, color='red', ha='left')

                ax1.set_title('Rejection/Non-rejection Region')
                ax1.set_xticks([0, t_crit])
                ax1.set_xlabel('$t$ Values')
                ax1.legend()

                # Save the combined figure
                plt.tight_layout()
                if not os.path.exists('static'):
                    os.makedirs('static')
                plt.savefig('static/plot_sigmaunknown.png')
                plt.close()

        except Exception as e:
            ci_result = f"Error: {e}"

    return render_template('popmean_sigmaunknown.html', values = values, CI = CI,test_type = test_type, df = df)






# Comparing two means:
@populationmean_bp.route("/twopopmeans", methods=["GET", "POST"])
def twopopmeans():
    values = None
    test_type = None
    CI = None
    df = None
    margin_error = None
    if request.method == 'POST':
        try:
            sample_mean1 = float(request.form['sample_mean1'])
            sample_mean2 = float(request.form['sample_mean2'])
            sample_size1 = int(request.form['sample_size1'])
            sample_size2 = int(request.form['sample_size2'])
            alpha = float(request.form['alpha'])
            sigma1 = float(request.form['s1'])
            sigma2 = float(request.form['s2'])
            df = int(request.form['df'])
            test_type = request.form['Type']
            s_pooled = np.sqrt(((sample_size1 - 1) * sigma1**2 + (sample_size2 - 1) * sigma2**2) / (sample_size1 + sample_size2 - 2))
            t_test = (sample_mean1 - sample_mean2) / (s_pooled * np.sqrt(1 / sample_size1 + 1 / sample_size2))
            if test_type == "two-tailed":
                t_crit = t.ppf(1 - alpha / 2, df)
                p_value = 2 * (1 - t.cdf(abs(t_test), df))
                margin_error = t_crit * s_pooled * np.sqrt(1 / sample_size1 + 1 / sample_size2)
                CI = (sample_mean1 - sample_mean2 - margin_error, sample_mean1 - sample_mean2 + margin_error)

                values = (sample_mean1, sample_mean2, sample_size1, sample_size2, sigma1, sigma2, alpha, t_test, t_crit, p_value)

                fig, ax1 = plt.subplots(1, 1, figsize=(6, 4))

                # ===== First Plot (Population Mean Distribution) =====
                x = np.linspace(-4, 4, 500)
                y = t.pdf(x, df)
                y1 = np.zeros(500)

                ax1.plot(x, y)
                ax1.plot(-t_crit, 0, 'ro', label = 'Critical Value')
                ax1.plot(t_crit, 0, 'ro', label = 'Critical Value')
                ax1.plot(t_test, 0, 'bo', label = 'T Test Statistic')
                ax1.plot(x, y1)


                # Fill rejection regions
                x_fill_left = np.linspace(-4, - t_crit, 500)
                y_fill_left = t.pdf(x_fill_left, df)

                x_fill_right = np.linspace(t_crit, 4, 500)
                y_fill_right = t.pdf(x_fill_right, df)

                ax1.fill_between(x_fill_left, y_fill_left, color='skyblue', alpha=0.6)
                ax1.fill_between(x_fill_right, y_fill_right, color='skyblue', alpha=0.6)

                ax1.axvline(-t_crit, color='red', linestyle='--')
                ax1.axvline(t_crit, color='red', linestyle='--')

                ax1.text(0, 0.1, 'Do Not Reject $H_0$', fontsize=10, color='red', ha='center')
                ax1.text(- t_crit - 0.1, 0.1, 'Reject $H_0$', fontsize=10, color='red', ha='right')
                ax1.text(t_crit + 0.1, 0.1, 'Reject $H_0$', fontsize=10, color='red', ha='left')

                ax1.set_title('Rejection/Non-rejection Region')
                ax1.set_xticks([-t_crit, 0, t_crit])
                ax1.set_xlabel('$t$ Values')
                ax1.legend()

                # Save the combined figure
                plt.tight_layout()
                if not os.path.exists('static'):
                    os.makedirs('static')
                plt.savefig('static/plot_twopopmeans.png')
                plt.close()

            elif test_type == "left-tailed":
                t_crit = t.ppf(alpha, df)
                p_value = t.cdf(t_test, df)

                values = (sample_mean1, sample_mean2, sample_size1, sample_size2, sigma1, sigma2, alpha, t_test, t_crit, p_value)
                margin_error = t_crit * s_pooled * np.sqrt(1 / sample_size1 + 1 / sample_size2)
                CI = (None, sample_mean1 - sample_mean2 + margin_error) #upper confidence bound

                # Create both subplots side by side
                fig, ax1 = plt.subplots(1, 1, figsize=(6, 4))

                # ===== First Plot (Population Mean Distribution) =====
                x = np.linspace(-4, 4, 500)
                y = t.pdf(x, df)
                y1 = np.zeros(500)

                ax1.plot(x, y)
                ax1.plot(t_crit, 0, 'ro', label = 'Critical Value')
        
                ax1.plot(t_test, 0, 'bo', label = 't Test Statistic')
                ax1.plot(x, y1)

              # Fill rejection regions
                x_fill_left = np.linspace(-4, t_crit, 500)
                y_fill_left = t.pdf(x_fill_left, df)

                ax1.fill_between(x_fill_left, y_fill_left, color='skyblue', alpha=0.6)

                ax1.axvline(t_crit, color='red', linestyle='--')

                ax1.text(t_crit - 0.15, 0.12, 'Reject $H_0$', fontsize=10, color='red', ha='right')
                ax1.text(t_crit + 0.15, 0.12, 'Do Not Reject $H_0$', fontsize=10, color='red', ha='left')

                ax1.set_title('Rejection/Non-rejection Region')
                ax1.set_xticks([t_crit, 0])
                ax1.set_xlabel('$t$ Values')
                ax1.legend()

                # Save the combined figure
                plt.tight_layout()
                if not os.path.exists('static'):
                    os.makedirs('static')
                plt.savefig('static/plot_twopopmeans.png')
                plt.close()
            elif test_type == "right-tailed":
                t_crit = t.ppf(1-alpha, df)
                p_value = 1 - t.cdf(t_test, df)
                margin_error = t_crit * s_pooled * np.sqrt(1 / sample_size1 + 1 / sample_size2)
                values = (sample_mean1, sample_mean2, sample_size1, sample_size2, sigma1, sigma2, alpha, t_test, t_crit, p_value)
                CI = (sample_mean1 - sample_mean2 - margin_error, None)

                # Create both subplots side by side
                fig, ax1 = plt.subplots(1, 1, figsize=(6, 4))

                # ===== First Plot (Population Mean Distribution) =====
                x = np.linspace(-4, 4, 500)
                y = t.pdf(x, df)
                y1 = np.zeros(500)

                ax1.plot(x, y)
                ax1.plot(t_crit, 0, 'ro', label = 'Critical Value')

                ax1.plot(t_test, 0, 'bo', label = 't Test Statistic')
                ax1.plot(x, y1)

              # Fill rejection regions
                x_fill_left = np.linspace(t_crit, 4, 500)
                y_fill_left = t.pdf(x_fill_left, df)

                ax1.fill_between(x_fill_left, y_fill_left, color='skyblue', alpha=0.6)

                ax1.axvline(t_crit, color='red', linestyle='--')

                ax1.text(t_crit - 0.15, 0.12, ' Do Not Reject $H_0$', fontsize=10, color='red', ha='right')
                ax1.text(t_crit + 0.15, 0.12, 'Reject $H_0$', fontsize=10, color='red', ha='left')

                ax1.set_title('Rejection/Non-rejection Region')
                ax1.set_xticks([0, t_crit])
                ax1.set_xlabel('$t$ Values')
                ax1.legend()

                # Save the combined figure
                plt.tight_layout()
                if not os.path.exists('static'):
                    os.makedirs('static')
                plt.savefig('static/plot_twopopmeans.png')
                plt.close()

        except Exception as e:
            ci_result = f"Error: {e}"

    return render_template('diffpopmeans.html', values = values, CI = CI,test_type = test_type, df = df, me = margin_error)








#comparing two means: unequal variances
@populationmean_bp.route("/diffpopmeans_unequalvar", methods=["GET", "POST"])
def diffpopmeans_unequalvar():
    values = None
    test_type = None
    CI = None
    df = None
    margin_error = None
    if request.method == 'POST':
        try:
            sample_mean1 = float(request.form['sample_mean1'])
            sample_mean2 = float(request.form['sample_mean2'])
            sample_size1 = int(request.form['sample_size1'])
            sample_size2 = int(request.form['sample_size2'])
            alpha = float(request.form['alpha'])
            sigma1 = float(request.form['s1'])
            sigma2 = float(request.form['s2'])
            test_type = request.form['Type']

            df = ( (sigma1**2/sample_size1 + sigma2**2/sample_size2)**2 ) / ( ((sigma1**2/sample_size1)**2)/(sample_size1-1) + ((sigma2**2/sample_size2)**2)/(sample_size2-1) )
            df = math.floor(df)
            t_test = (sample_mean1 - sample_mean2) / np.sqrt(sigma1**2 / sample_size1 + sigma2**2 / sample_size2)
            if test_type == "two-tailed":
                t_crit = t.ppf(1 - alpha / 2, df)
                p_value = 2 * (1 - t.cdf(abs(t_test), df))
                margin_error = t_crit * np.sqrt(sigma1**2 / sample_size1 + sigma2**2 / sample_size2)
                CI = (sample_mean1 - sample_mean2 - margin_error, sample_mean1 - sample_mean2 + margin_error)

                values = (sample_mean1, sample_mean2, sample_size1, sample_size2, sigma1, sigma2, alpha, t_test, t_crit, p_value)

                fig, ax1 = plt.subplots(1, 1, figsize=(6, 4))

                # ===== First Plot (Population Mean Distribution) =====
                x = np.linspace(-4, 4, 500)
                y = t.pdf(x, df)
                y1 = np.zeros(500)

                ax1.plot(x, y)
                ax1.plot(-t_crit, 0, 'ro', label = 'Critical Value')
                ax1.plot(t_crit, 0, 'ro', label = 'Critical Value')
                ax1.plot(t_test, 0, 'bo', label = 'T Test Statistic')
                ax1.plot(x, y1)


                # Fill rejection regions
                x_fill_left = np.linspace(-4, - t_crit, 500)
                y_fill_left = t.pdf(x_fill_left, df)

                x_fill_right = np.linspace(t_crit, 4, 500)
                y_fill_right = t.pdf(x_fill_right, df)

                ax1.fill_between(x_fill_left, y_fill_left, color='skyblue', alpha=0.6)
                ax1.fill_between(x_fill_right, y_fill_right, color='skyblue', alpha=0.6)

                ax1.axvline(-t_crit, color='red', linestyle='--')
                ax1.axvline(t_crit, color='red', linestyle='--')

                ax1.text(0, 0.1, 'Do Not Reject $H_0$', fontsize=10, color='red', ha='center')
                ax1.text(- t_crit - 0.1, 0.1, 'Reject $H_0$', fontsize=10, color='red', ha='right')
                ax1.text(t_crit + 0.1, 0.1, 'Reject $H_0$', fontsize=10, color='red', ha='left')

                ax1.set_title('Rejection/Non-rejection Region')
                ax1.set_xticks([-t_crit, 0, t_crit])
                ax1.set_xlabel('$t$ Values')
                ax1.legend()

                # Save the combined figure
                plt.tight_layout()
                if not os.path.exists('static'):
                    os.makedirs('static')
                plt.savefig('static/plot_twopopmeans_uneqvar.png')
                plt.close()

            elif test_type == "left-tailed":
                t_crit = t.ppf(alpha, df)
                p_value = t.cdf(t_test, df)

                values = (sample_mean1, sample_mean2, sample_size1, sample_size2, sigma1, sigma2, alpha, t_test, t_crit, p_value)
                margin_error = t_crit *  np.sqrt(sigma1**2 / sample_size1 + sigma2**2 / sample_size2)
                CI = (None, sample_mean1 - sample_mean2 + margin_error) #upper confidence bound

                # Create both subplots side by side
                fig, ax1 = plt.subplots(1, 1, figsize=(6, 4))

                # ===== First Plot (Population Mean Distribution) =====
                x = np.linspace(-4, 4, 500)
                y = t.pdf(x, df)
                y1 = np.zeros(500)

                ax1.plot(x, y)
                ax1.plot(t_crit, 0, 'ro', label = 'Critical Value')
        
                ax1.plot(t_test, 0, 'bo', label = 't Test Statistic')
                ax1.plot(x, y1)

              # Fill rejection regions
                x_fill_left = np.linspace(-4, t_crit, 500)
                y_fill_left = t.pdf(x_fill_left, df)

                ax1.fill_between(x_fill_left, y_fill_left, color='skyblue', alpha=0.6)

                ax1.axvline(t_crit, color='red', linestyle='--')

                ax1.text(t_crit - 0.15, 0.12, 'Reject $H_0$', fontsize=10, color='red', ha='right')
                ax1.text(t_crit + 0.15, 0.12, 'Do Not Reject $H_0$', fontsize=10, color='red', ha='left')

                ax1.set_title('Rejection/Non-rejection Region')
                ax1.set_xticks([t_crit, 0])
                ax1.set_xlabel('$t$ Values')
                ax1.legend()

                # Save the combined figure
                plt.tight_layout()
                if not os.path.exists('static'):
                    os.makedirs('static')
                plt.savefig('static/plot_twopopmeans_uneqvar.png')
                plt.close()
            elif test_type == "right-tailed":
                t_crit = t.ppf(1-alpha, df)
                p_value = 1 - t.cdf(t_test, df)
                margin_error = t_crit * np.sqrt(sigma1**2 / sample_size1 + sigma2**2 / sample_size2)
                values = (sample_mean1, sample_mean2, sample_size1, sample_size2, sigma1, sigma2, alpha, t_test, t_crit, p_value)
                CI = (sample_mean1 - sample_mean2 - margin_error, None)

                # Create both subplots side by side
                fig, ax1 = plt.subplots(1, 1, figsize=(6, 4))

                # ===== First Plot (Population Mean Distribution) =====
                x = np.linspace(-4, 4, 500)
                y = t.pdf(x, df)
                y1 = np.zeros(500)

                ax1.plot(x, y)
                ax1.plot(t_crit, 0, 'ro', label = 'Critical Value')

                ax1.plot(t_test, 0, 'bo', label = 't Test Statistic')
                ax1.plot(x, y1)

              # Fill rejection regions
                x_fill_left = np.linspace(t_crit, 4, 500)
                y_fill_left = t.pdf(x_fill_left, df)

                ax1.fill_between(x_fill_left, y_fill_left, color='skyblue', alpha=0.6)

                ax1.axvline(t_crit, color='red', linestyle='--')

                ax1.text(t_crit - 0.15, 0.12, ' Do Not Reject $H_0$', fontsize=10, color='red', ha='right')
                ax1.text(t_crit + 0.15, 0.12, 'Reject $H_0$', fontsize=10, color='red', ha='left')

                ax1.set_title('Rejection/Non-rejection Region')
                ax1.set_xticks([0, t_crit])
                ax1.set_xlabel('$t$ Values')
                ax1.legend()

                # Save the combined figure
                plt.tight_layout()
                if not os.path.exists('static'):
                    os.makedirs('static')
                plt.savefig('static/plot_twopopmeans_uneqvar.png')
                plt.close()

        except Exception as e:
            ci_result = f"Error: {e}"

    return render_template('diffpopmeans_unequalvar.html', values = values, CI = CI,test_type = test_type, df = df, me = margin_error)