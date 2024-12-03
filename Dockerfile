# Use the official Python 3.10 image as a base
FROM python:3.10

# Set the working directory in the container
WORKDIR /app

# Copy the pyproject.toml and poetry.lock files to the working directory
COPY pyproject.toml poetry.lock* ./

# Install Poetry
RUN pip install poetry

# Install the dependencies defined in pyproject.toml
RUN poetry install --no-root

# Copy the rest of your application code to the working directory
COPY . .

# Expose the ports for Jupyter and Streamlit
EXPOSE 8888 8501

# Command to run Jupyter notebook and Streamlit app
CMD ["sh", "-c", "poetry run jupyter notebook --ip=0.0.0.0 --port=8888 --no-browser --allow-root & poetry run streamlit run streamlit_app/home.py"]