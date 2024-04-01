from flask import Flask, render_template, request, redirect, url_for
import csv
import pandas as pd

app = Flask(__name__)

# Read data from CSV
book_data = pd.read_csv('goodreads.csv')

# Define the path to the CSV file
CSV_FILE = 'ratings_comments.csv'

# Function to append data to the CSV file
def append_to_csv(data):
    with open(CSV_FILE, 'a', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=['Rating', 'Comment'])
        writer.writerow(data)

@app.route('/')
def index():
    # Select a random subset of books
    random_books = book_data.sample(n=8)
    return render_template('index.html', random_books=random_books.to_dict('records'))

@app.route('/book/<isbn>', methods=['GET', 'POST'])
def book(isbn):
    book_info = book_data[book_data['isbn'] == isbn].iloc[0]
    if request.method == 'POST':
        rating = request.form.get('rating')
        comment = request.form.get('comment')
        if rating and comment:
            # Append rating and comment to CSV file
            append_to_csv({'Rating': rating, 'Comment': comment})
        # Redirect to book details page
        return redirect(url_for('book', isbn=isbn))
    return render_template('book.html', book=book_info.to_dict())

if __name__ == '__main__':
    app.run(debug=True)
