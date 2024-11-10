import random
import csv

# Function to generate a random entry
def generate_entry():
    profile_pic = random.randint(0, 1)
    username_length = round(random.uniform(0, 1), 2)
    fullname_words = random.randint(1, 5)
    fullname_length = round(random.uniform(0, 1), 2)
    name_equals_username = random.randint(0, 1)
    description_length = random.randint(0, 150)
    external_url = random.randint(0, 1)
    private = random.randint(0, 1)
    num_posts = random.randint(0, 1000)
    num_followers = random.randint(0, 100000)
    num_follows = random.randint(0, 100000)
    is_fake = random.randint(0, 1)

    return [
        profile_pic, username_length, fullname_words, fullname_length, name_equals_username,
        description_length, external_url, private, num_posts, num_followers, num_follows, is_fake
    ]

# Generate 100 entries
data = [generate_entry() for _ in range(100)]

# Write to a new CSV file
with open('generated_data.csv', 'w', newline='') as csvfile:
    csvwriter = csv.writer(csvfile)
    csvwriter.writerow([
        "profile pic", "nums/length username", "fullname words", "nums/length fullname",
        "name==username", "description length", "external URL", "private",
        "#posts", "#followers", "#follows", "fake"
    ])
    csvwriter.writerows(data)

print("Generated data saved to 'generated_data.csv'")
