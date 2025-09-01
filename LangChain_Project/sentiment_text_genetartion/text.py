# Define the dictionary before using it
my_dict = {'text': 'Hello, world!'}

# Using 'in' operator
if 'text' in my_dict:
    print(my_dict['text'])
else:
    print("Key 'text' not found.")

    # Using get() method (returns None or a default value if key is not found)
    value = my_dict.get('text', "Default Text")
    print(value)