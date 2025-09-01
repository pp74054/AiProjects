import pandas as pd

df = pd.DataFrame({
    'numbers': [1, 2, 3],
    'date': ['2023-01-05', '2023-03-25', '2023-01-24']
})

arr = ['2023-01-05']

print(df.iloc[0]['date'])  # 👉️ "2023-01-05"

if df.iloc[0]['date'] == arr[0]:
    # 👇️ this runs
    print('success')
else:
    print('failure')
