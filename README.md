# Autoregressive Image Generation with MADE 🚀

Welcome to the Autoregressive Image Generation with MADE project! This repository implements a Masked Autoencoder for Distribution Estimation (MADE) to learn and generate samples from synthetic 2D datasets like moons, blobs, and spirals, inspired by Germain et al. (2015). 📊 Dive into autoregressive modeling, visualize stunning scatter plots, and explore how network depth impacts generative quality—all in a GitHub Codespace environment! 🌟
🎯 Project Overview
This project trains a MADE model to capture the joint distribution of 2D toy datasets, generates new samples, and visualizes them alongside the originals. It also experiments with shallow and deep network architectures to study their effect on sample quality. Built with PyTorch and scikit-learn, it’s perfect for exploring generative modeling in a simple yet powerful way.
Key Objectives:

Train MADE on datasets like moons, blobs, and spirals.
Generate and visualize samples as 2D scatter plots.
Compare shallow ([64, 64]) and deep ([128, 128, 128]) network performance.
Save training loss curves for analysis.

🛠️ Setup in GitHub Codespace
Get started in minutes with these steps:

Clone or Create Repository 📂

Clone this repository or create a new one on GitHub.
Ensure it contains made_project.py, requirements.txt, and this README.md.


Launch Codespace ☁️

Open your repository on GitHub.
Click Code > Codespaces > Create Codespace on main.


Install Dependencies 🛠️

In the Codespace terminal, run:pip install -r requirements.txt


This installs torch, numpy, scikit-learn, and matplotlib.



🚀 Running the Project

Execute the Script 🖥️

Run the main script in the terminal:python made_project.py


The script trains the MADE model on three datasets (moons, blobs, spirals) with two configurations (shallow and deep).


What to Expect 📈

Training progress is printed every 20 epochs, showing the loss.
PNG files are generated for scatter plots and loss curves.


View Outputs 🖼️

In Codespace’s file explorer, right-click a PNG file (e.g., moons_Shallow_(64,_64).png) and select Open in Preview.



📁 Project Structure



File
Description



made_project.py
Core script for dataset generation, MADE model, training, and visualization.


requirements.txt
Lists Python dependencies (torch, numpy, etc.).


README.md
This file, guiding you through setup, execution, and outputs.


📊 Outputs
The project generates two types of visualizations:

Scatter Plots 📍

Compare original (blue) and generated (red) samples.
Examples: moons_Shallow_(64,_64).png, spirals_Deep_(128,_128,_128).png.
Each plot shows side-by-side scatter plots for qualitative comparison.


Loss Curves 📉

Show training loss over 100 epochs.
Examples: loss_blobs_Shallow_(64,_64).png, loss_spirals_Deep_(128,_128,_128).png.



Find these files in your Codespace working directory and preview them directly!
🧩 Requirements
The project depends on the following packages (see requirements.txt):

torch>=1.9.0
numpy>=1.19.0
scikit-learn>=0.24.0
matplotlib>=3.3.0

💡 Tips & Notes

Customize ✨: Adjust hyperparameters in made_project.py (e.g., learning rate, epochs) to improve sample quality.
Extend 🔍: Add test set likelihood computation by splitting datasets and calculating log-likelihood in train_made.
Troubleshooting ⚠️: If errors occur, verify dependencies with pip list. Check mask shapes in the MADE model for dimension issues.
Reference 📚: Germain, M., et al. (2015). MADE: Masked Autoencoder for Distribution Estimation.
Feedback 💬: Found an issue or have a suggestion? Open an issue in the repository!

🌟 Get Involved
Explore the power of autoregressive modeling and share your results! Fork this repo, tweak the model, or contribute improvements. Happy coding! 🎉
