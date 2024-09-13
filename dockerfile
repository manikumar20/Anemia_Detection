# Use Python 3.12.3 as the base image
FROM python:3.12.3

# Set the working directory in the container
WORKDIR /app1

# Copy the current directory contents into the container at /app
COPY . /app1

# Install any needed dependencies specified in requirements.txt
RUN pip install --no-cache-dir pandas numpy scikit-learn==1.4.1.post1 Flask joblib tensorflow==2.16.1

# Make port 5000 available to the world outside this container
EXPOSE 5000

# Define environment variable
ENV FLASK_APP=app1.py

# Run the Flask application when the container launches
CMD ["flask", "run", "--host=0.0.0.0"]
