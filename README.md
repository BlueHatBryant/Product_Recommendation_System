# Product Recommendation System

## Overview
This is a simple product recommendation system built with Streamlit. The system allows users to select a product by ID and receive suggestions for related items based on text similarity. It uses TF-IDF vectorization and cosine similarity to identify related products from a dataset.

## Features
- Select a product ID from a dropdown menu.
- View a short description of the selected product (first three sentences).
- Expand to see the full product description.
- Receive up to three related product recommendations.

## Requirements
The project requires the following Python packages:
```
streamlit
pandas
scikit-learn
```

## Dataset
The application uses a CSV file named `sample-data.csv`, which contains two columns:
- `id`: Product ID
- `description`: Full product description (including HTML tags)

Ensure that `sample-data.csv` is in the same directory as `app.py`.

## License
This project is licensed under the Apache License 2.0. See the `LICENSE` file for details.