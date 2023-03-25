import random
import pandas as pd
from typing import List
from faker import Faker


fake = Faker()


def generate_fake_data(num_rows=10):
    data = []
    industries = ['Automotive', 'Technology', 'Healthcare', 'Finance', 'Retail']
    sectors = ['Manufacturing', 'Services', 'Software', 'Banking', 'Logistics']

    for _ in range(num_rows):
        company_name = fake.company()
        total_value = round(random.uniform(100000, 10000000), 2)
        total_cost = round(random.uniform(50000, 5000000), 2)
        country = fake.country()
        industry = random.choice(industries)
        sector = random.choice(sectors)

        row = {
            'company_name': company_name,
            'total_value': total_value,
            'total_cost': total_cost,
            'country': country,
            'industry': industry,
            'sector': sector
        }

        data.append(row)

    return pd.DataFrame(data)


def table_preprocesser(tables: List|pd.DataFrame):
    tables = [tables] if type(tables) == pd.DataFrame else tables

    processed_tables = ["\n".join([table.to_csv(index=False)])
                        for table in tables]

    return processed_tables


if __name__ == "__main__":
    num_rows = 200
    fake_data = generate_fake_data(num_rows)
